import logging
import os
import platform
import socket
import string
from base64 import b64encode
from urllib import parse
import certifi
import urllib3
from selenium import __version__
from . import utils
from .command import Command
from .errorhandler import ErrorCode
class RemoteConnection:
    """A connection with the Remote WebDriver server.

    Communicates with the server using the WebDriver wire protocol:
    https://github.com/SeleniumHQ/selenium/wiki/JsonWireProtocol
    """
    browser_name = None
    _timeout = socket._GLOBAL_DEFAULT_TIMEOUT
    _ca_certs = os.getenv('REQUESTS_CA_BUNDLE') if 'REQUESTS_CA_BUNDLE' in os.environ else certifi.where()

    @classmethod
    def get_timeout(cls):
        """:Returns:

        Timeout value in seconds for all http requests made to the
        Remote Connection
        """
        return None if cls._timeout == socket._GLOBAL_DEFAULT_TIMEOUT else cls._timeout

    @classmethod
    def set_timeout(cls, timeout):
        """Override the default timeout.

        :Args:
            - timeout - timeout value for http requests in seconds
        """
        cls._timeout = timeout

    @classmethod
    def reset_timeout(cls):
        """Reset the http request timeout to socket._GLOBAL_DEFAULT_TIMEOUT."""
        cls._timeout = socket._GLOBAL_DEFAULT_TIMEOUT

    @classmethod
    def get_certificate_bundle_path(cls):
        """:Returns:

        Paths of the .pem encoded certificate to verify connection to
        command executor. Defaults to certifi.where() or
        REQUESTS_CA_BUNDLE env variable if set.
        """
        return cls._ca_certs

    @classmethod
    def set_certificate_bundle_path(cls, path):
        """Set the path to the certificate bundle to verify connection to
        command executor. Can also be set to None to disable certificate
        validation.

        :Args:
            - path - path of a .pem encoded certificate chain.
        """
        cls._ca_certs = path

    @classmethod
    def get_remote_connection_headers(cls, parsed_url, keep_alive=False):
        """Get headers for remote request.

        :Args:
         - parsed_url - The parsed url
         - keep_alive (Boolean) - Is this a keep-alive connection (default: False)
        """
        system = platform.system().lower()
        if system == 'darwin':
            system = 'mac'
        headers = {'Accept': 'application/json', 'Content-Type': 'application/json;charset=UTF-8', 'User-Agent': f'selenium/{__version__} (python {system})'}
        if parsed_url.username:
            base64string = b64encode(f'{parsed_url.username}:{parsed_url.password}'.encode())
            headers.update({'Authorization': f'Basic {base64string.decode()}'})
        if keep_alive:
            headers.update({'Connection': 'keep-alive'})
        return headers

    def _get_proxy_url(self):
        if self._url.startswith('https://'):
            return os.environ.get('https_proxy', os.environ.get('HTTPS_PROXY'))
        if self._url.startswith('http://'):
            return os.environ.get('http_proxy', os.environ.get('HTTP_PROXY'))

    def _identify_http_proxy_auth(self):
        url = self._proxy_url
        url = url[url.find(':') + 3:]
        return '@' in url and len(url[:url.find('@')]) > 0

    def _separate_http_proxy_auth(self):
        url = self._proxy_url
        protocol = url[:url.find(':') + 3]
        no_protocol = url[len(protocol):]
        auth = no_protocol[:no_protocol.find('@')]
        proxy_without_auth = protocol + no_protocol[len(auth) + 1:]
        return (proxy_without_auth, auth)

    def _get_connection_manager(self):
        pool_manager_init_args = {'timeout': self.get_timeout()}
        if self._ca_certs:
            pool_manager_init_args['cert_reqs'] = 'CERT_REQUIRED'
            pool_manager_init_args['ca_certs'] = self._ca_certs
        if self._proxy_url:
            if self._proxy_url.lower().startswith('sock'):
                from urllib3.contrib.socks import SOCKSProxyManager
                return SOCKSProxyManager(self._proxy_url, **pool_manager_init_args)
            if self._identify_http_proxy_auth():
                self._proxy_url, self._basic_proxy_auth = self._separate_http_proxy_auth()
                pool_manager_init_args['proxy_headers'] = urllib3.make_headers(proxy_basic_auth=self._basic_proxy_auth)
            return urllib3.ProxyManager(self._proxy_url, **pool_manager_init_args)
        return urllib3.PoolManager(**pool_manager_init_args)

    def __init__(self, remote_server_addr: str, keep_alive: bool=False, ignore_proxy: bool=False):
        self.keep_alive = keep_alive
        self._url = remote_server_addr
        _no_proxy = os.environ.get('no_proxy', os.environ.get('NO_PROXY'))
        if _no_proxy:
            for npu in _no_proxy.split(','):
                npu = npu.strip()
                if npu == '*':
                    ignore_proxy = True
                    break
                n_url = parse.urlparse(npu)
                remote_add = parse.urlparse(self._url)
                if n_url.netloc:
                    if remote_add.netloc == n_url.netloc:
                        ignore_proxy = True
                        break
                elif n_url.path in remote_add.netloc:
                    ignore_proxy = True
                    break
        self._proxy_url = self._get_proxy_url() if not ignore_proxy else None
        if keep_alive:
            self._conn = self._get_connection_manager()
        self._commands = remote_commands

    def execute(self, command, params):
        """Send a command to the remote server.

        Any path substitutions required for the URL mapped to the command should be
        included in the command parameters.

        :Args:
         - command - A string specifying the command to execute.
         - params - A dictionary of named parameters to send with the command as
           its JSON payload.
        """
        command_info = self._commands[command]
        assert command_info is not None, f'Unrecognised command {command}'
        path_string = command_info[1]
        path = string.Template(path_string).substitute(params)
        substitute_params = {word[1:] for word in path_string.split('/') if word.startswith('$')}
        if isinstance(params, dict) and substitute_params:
            for word in substitute_params:
                del params[word]
        data = utils.dump_json(params)
        url = f'{self._url}{path}'
        trimmed = self._trim_large_entries(params)
        LOGGER.debug('%s %s %s', command_info[0], url, str(trimmed))
        return self._request(command_info[0], url, body=data)

    def _request(self, method, url, body=None):
        """Send an HTTP request to the remote server.

        :Args:
         - method - A string for the HTTP method to send the request with.
         - url - A string for the URL to send the request to.
         - body - A string for request body. Ignored unless method is POST or PUT.

        :Returns:
          A dictionary with the server's parsed JSON response.
        """
        parsed_url = parse.urlparse(url)
        headers = self.get_remote_connection_headers(parsed_url, self.keep_alive)
        response = None
        if body and method not in ('POST', 'PUT'):
            body = None
        if self.keep_alive:
            response = self._conn.request(method, url, body=body, headers=headers)
            statuscode = response.status
        else:
            conn = self._get_connection_manager()
            with conn as http:
                response = http.request(method, url, body=body, headers=headers)
            statuscode = response.status
        data = response.data.decode('UTF-8')
        LOGGER.debug('Remote response: status=%s | data=%s | headers=%s', response.status, data, response.headers)
        try:
            if 300 <= statuscode < 304:
                return self._request('GET', response.headers.get('location', None))
            if 399 < statuscode <= 500:
                return {'status': statuscode, 'value': data}
            content_type = []
            if response.headers.get('Content-Type', None):
                content_type = response.headers.get('Content-Type', None).split(';')
            if not any([x.startswith('image/png') for x in content_type]):
                try:
                    data = utils.load_json(data.strip())
                except ValueError:
                    if 199 < statuscode < 300:
                        status = ErrorCode.SUCCESS
                    else:
                        status = ErrorCode.UNKNOWN_ERROR
                    return {'status': status, 'value': data.strip()}
                if 'value' not in data:
                    data['value'] = None
                return data
            data = {'status': 0, 'value': data}
            return data
        finally:
            LOGGER.debug('Finished Request')
            response.close()

    def close(self):
        """Clean up resources when finished with the remote_connection."""
        if hasattr(self, '_conn'):
            self._conn.clear()

    def _trim_large_entries(self, input_dict, max_length=100):
        """Truncate string values in a dictionary if they exceed max_length.

        :param dict: Dictionary with potentially large values
        :param max_length: Maximum allowed length of string values
        :return: Dictionary with truncated string values
        """
        output_dictionary = {}
        for key, value in input_dict.items():
            if isinstance(value, dict):
                output_dictionary[key] = self._trim_large_entries(value, max_length)
            elif isinstance(value, str) and len(value) > max_length:
                output_dictionary[key] = value[:max_length] + '...'
            else:
                output_dictionary[key] = value
        return output_dictionary