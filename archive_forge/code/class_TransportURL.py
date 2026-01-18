import logging
from debtcollector import removals
from oslo_config import cfg
from stevedore import driver
from urllib import parse
from oslo_messaging import exceptions
class TransportURL(object):
    """A parsed transport URL.

    Transport URLs take the form::

      driver://[user:pass@]host:port[,[userN:passN@]hostN:portN]/virtual_host?query

    where:

    driver
      Specifies the transport driver to use. Typically this is `rabbit` for the
      RabbitMQ broker. See the documentation for other available transport
      drivers.

    [user:pass@]host:port
      Specifies the network location of the broker. `user` and `pass` are the
      optional username and password used for authentication with the broker.

      `user` and `pass` may contain any of the following ASCII characters:
        * Alphabetic (a-z and A-Z)
        * Numeric (0-9)
        * Special characters: & = $ - _ . + ! * ( )

      `user` may include at most one `@` character for compatibility with some
      implementations of SASL.

      All other characters in `user` and `pass` must be encoded via '%nn'

      You may include multiple different network locations separated by commas.
      The client will connect to any of the available locations and will
      automatically fail over to another should the connection fail.

    virtual_host
      Specifies the "virtual host" within the broker. Support for virtual hosts
      is specific to the message bus used.

    query
      Permits passing driver-specific options which override the corresponding
      values from the configuration file.

    :param conf: a ConfigOpts instance
    :type conf: oslo.config.cfg.ConfigOpts
    :param transport: a transport name for example 'rabbit'
    :type transport: str
    :param virtual_host: a virtual host path for example '/'
    :type virtual_host: str
    :param hosts: a list of TransportHost objects
    :type hosts: list
    :param query: a dictionary of URL query parameters
    :type query: dict
    """

    def __init__(self, conf, transport=None, virtual_host=None, hosts=None, query=None):
        self.conf = conf
        self.conf.register_opts(_transport_opts)
        self.transport = transport
        self.virtual_host = virtual_host
        if hosts is None:
            self.hosts = []
        else:
            self.hosts = hosts
        if query is None:
            self.query = {}
        else:
            self.query = query

    def __hash__(self):
        return hash((tuple(self.hosts), self.transport, self.virtual_host))

    def __eq__(self, other):
        return self.transport == other.transport and self.virtual_host == other.virtual_host and (self.hosts == other.hosts)

    def __ne__(self, other):
        return not self == other

    def __repr__(self):
        attrs = []
        for a in ['transport', 'virtual_host', 'hosts']:
            v = getattr(self, a)
            if v:
                attrs.append((a, repr(v)))
        values = ', '.join(['%s=%s' % i for i in attrs])
        return '<TransportURL ' + values + '>'

    def __str__(self):
        netlocs = []
        for host in self.hosts:
            username = host.username
            password = host.password
            hostname = host.hostname
            port = host.port
            netloc = ''
            if username is not None or password is not None:
                if username is not None:
                    netloc += parse.quote(username, '')
                if password is not None:
                    netloc += ':%s' % parse.quote(password, '')
                netloc += '@'
            if hostname:
                if ':' in hostname:
                    netloc += '[%s]' % hostname
                else:
                    netloc += hostname
            if port is not None:
                netloc += ':%d' % port
            netlocs.append(netloc)
        url = '%s://%s/' % (self.transport, ','.join(netlocs))
        if self.virtual_host:
            url += parse.quote(self.virtual_host)
        if self.query:
            url += '?' + parse.urlencode(self.query, doseq=True)
        return url

    @classmethod
    def parse(cls, conf, url=None):
        """Parse a URL as defined by :py:class:`TransportURL` and return a
        TransportURL object.

        Assuming a URL takes the form of::

          transport://user:pass@host:port[,userN:passN@hostN:portN]/virtual_host?query

        then parse the URL and return a TransportURL object.

        Netloc is parsed following the sequence bellow:

        * It is first split by ',' in order to support multiple hosts
        * All hosts should be specified with username/password or not
          at the same time. In case of lack of specification, username and
          password will be omitted::

            user:pass@host1:port1,host2:port2

            [
              {"username": "user", "password": "pass", "host": "host1:port1"},
              {"host": "host2:port2"}
            ]

        If the url is not provided conf.transport_url is parsed instead.

        :param conf: a ConfigOpts instance
        :type conf: oslo.config.cfg.ConfigOpts
        :param url: The URL to parse
        :type url: str
        :returns: A TransportURL
        """
        if not url:
            conf.register_opts(_transport_opts)
        url = url or conf.transport_url
        if not isinstance(url, str):
            raise InvalidTransportURL(url, 'Wrong URL type')
        url = parse.urlparse(url)
        if not url.scheme:
            raise InvalidTransportURL(url.geturl(), 'No scheme specified')
        transport = url.scheme
        query = {}
        if url.query:
            for key, values in parse.parse_qs(url.query).items():
                query[key] = ','.join(values)
        virtual_host = None
        if url.path.startswith('/'):
            virtual_host = parse.unquote(url.path[1:])
        hosts_with_credentials = []
        hosts_without_credentials = []
        hosts = []
        for host in url.netloc.split(','):
            if not host:
                continue
            hostname = host
            username = password = port = None
            if '@' in host:
                username, hostname = host.rsplit('@', 1)
                if ':' in username:
                    username, password = username.split(':', 1)
                    password = parse.unquote(password)
                username = parse.unquote(username)
            if not hostname:
                hostname = None
            elif hostname.startswith('['):
                host_end = hostname.find(']')
                if host_end < 0:
                    raise ValueError('Invalid IPv6 URL')
                port_text = hostname[host_end:]
                hostname = hostname[1:host_end]
                port = None
                if ':' in port_text:
                    port = port_text.split(':', 1)[1]
            elif ':' in hostname:
                hostname, port = hostname.split(':', 1)
            if port == '':
                port = None
            if port is not None:
                port = int(port)
            if username is None or password is None:
                hosts_without_credentials.append(hostname)
            else:
                hosts_with_credentials.append(hostname)
            hosts.append(TransportHost(hostname=hostname, port=port, username=username, password=password))
        if len(hosts_with_credentials) > 0 and len(hosts_without_credentials) > 0:
            LOG.warning('All hosts must be set with username/password or not at the same time. Hosts with credentials are: %(hosts_with_credentials)s. Hosts without credentials are %(hosts_without_credentials)s.', {'hosts_with_credentials': hosts_with_credentials, 'hosts_without_credentials': hosts_without_credentials})
        return cls(conf, transport, virtual_host, hosts, query)