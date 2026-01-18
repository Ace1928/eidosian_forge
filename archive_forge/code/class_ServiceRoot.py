from email.message import Message
from io import BytesIO
from json import dumps, loads
import sys
from wadllib.application import Resource as WadlResource
from lazr.restfulclient import __version__
from lazr.restfulclient._browser import Browser, RestfulHttp
from lazr.restfulclient._json import DatetimeJSONEncoder
from lazr.restfulclient.errors import HTTPError
from lazr.uri import URI
class ServiceRoot(Resource):
    """Entry point to the service. Subclass this for a service-specific client.

    :ivar credentials: The credentials instance used to access Launchpad.
    """
    RESOURCE_TYPE_CLASSES = {'HostedFile': HostedFile, 'ScalarValue': ScalarValue}

    def __init__(self, authorizer, service_root, cache=None, timeout=None, proxy_info=None, version=None, base_client_name='', max_retries=Browser.MAX_RETRIES):
        """Root access to a lazr.restful API.

        :param credentials: The credentials used to access the service.
        :param service_root: The URL to the root of the web service.
        :type service_root: string
        """
        if version is not None:
            if service_root[-1] != '/':
                service_root += '/'
            service_root += str(version)
            if service_root[-1] != '/':
                service_root += '/'
        self._root_uri = URI(service_root)
        self._base_client_name = base_client_name
        self.credentials = authorizer
        self._browser = Browser(self, authorizer, cache, timeout, proxy_info, self._user_agent, max_retries)
        self._wadl = self._browser.get_wadl_application(self._root_uri)
        root_resource = self._wadl.get_resource_by_path('')
        bound_root = root_resource.bind(self._browser.get(root_resource), 'application/json')
        super(ServiceRoot, self).__init__(None, bound_root)

    @property
    def _user_agent(self):
        """The value for the User-Agent header.

        This will be something like:
        launchpadlib 1.6.1, lazr.restfulclient 1.0.0; application=apport

        That is, a string describing lazr.restfulclient and an
        optional custom client built on top, and parameters containing
        any authorization-specific information that identifies the
        user agent (such as the application name).
        """
        base_portion = 'lazr.restfulclient %s' % __version__
        if self._base_client_name != '':
            base_portion = self._base_client_name + ' (' + base_portion + ')'
        message = Message()
        message['User-Agent'] = base_portion
        if self.credentials is not None:
            user_agent_params = self.credentials.user_agent_params
            for key in sorted(user_agent_params):
                value = user_agent_params[key]
                message.set_param(key, value, 'User-Agent')
        return message['User-Agent']

    def httpFactory(self, authorizer, cache, timeout, proxy_info):
        return RestfulHttp(authorizer, cache, timeout, proxy_info)

    def load(self, url):
        """Load a resource given its URL."""
        parsed = urlparse(url)
        if parsed.scheme == '':
            if url[:1] == '/':
                url = url[1:]
            url = str(self._root_uri.append(url))
        document = self._browser.get(url)
        if isinstance(document, binary_type):
            document = document.decode('utf-8')
        try:
            representation = loads(document)
        except ValueError:
            raise ValueError("%s doesn't serve a JSON document." % url)
        type_link = representation.get('resource_type_link')
        if type_link is None:
            raise ValueError("Couldn't determine the resource type of %s." % url)
        resource_type = self._root._wadl.get_resource_type(type_link)
        wadl_resource = WadlResource(self._root._wadl, url, resource_type.tag)
        return self._create_bound_resource(self._root, wadl_resource, representation, 'application/json', representation_needs_processing=False)