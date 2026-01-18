class RedirectLocation(_XMLKeyValue):
    """Specify redirect behavior for every request to a bucket's endpoint.

    :ivar hostname: Name of the host where requests will be redirected.

    :ivar protocol: Protocol to use (http, https) when redirecting requests.
        The default is the protocol that is used in the original request.

    """
    TRANSLATOR = [('HostName', 'hostname'), ('Protocol', 'protocol')]

    def __init__(self, hostname=None, protocol=None):
        self.hostname = hostname
        self.protocol = protocol
        super(RedirectLocation, self).__init__(self.TRANSLATOR)

    def to_xml(self):
        return tag('RedirectAllRequestsTo', super(RedirectLocation, self).to_xml())