import webob
from oslo_config import cfg
from oslo_log import log as logging
from oslo_serialization import jsonutils
from oslo_utils import strutils
import requests
class S3Token(object):
    """Middleware that handles S3 authentication."""

    def __init__(self, app, conf):
        """Common initialization code."""
        self._app = app
        self._logger = logging.getLogger(conf.get('log_name', __name__))
        self._logger.debug('Starting the %s component', PROTOCOL_NAME)
        self._reseller_prefix = conf.get('reseller_prefix', 'AUTH_')
        self._request_uri = conf.get('www_authenticate_uri')
        auth_uri = conf.get('auth_uri')
        if not self._request_uri and auth_uri:
            self._logger.warning('Use of the auth_uri option was deprecated in the Queens release in favor of www_authenticate_uri. This option will be removed in the S release.')
            self._request_uri = auth_uri
        if not self._request_uri:
            self._logger.warning('Use of the auth_host, auth_port, and auth_protocol configuration options was deprecated in the Newton release in favor of www_authenticate_uri. These options will be removed in the S release.')
            auth_host = conf.get('auth_host')
            auth_port = int(conf.get('auth_port', 35357))
            auth_protocol = conf.get('auth_protocol', 'https')
            self._request_uri = '%s://%s:%s' % (auth_protocol, auth_host, auth_port)
        insecure = strutils.bool_from_string(conf.get('insecure', False))
        cert_file = conf.get('certfile')
        key_file = conf.get('keyfile')
        if insecure:
            self._verify = False
        elif cert_file and key_file:
            self._verify = (cert_file, key_file)
        elif cert_file:
            self._verify = cert_file
        else:
            self._verify = None

    def _deny_request(self, code):
        error_table = {'AccessDenied': (401, 'Access denied'), 'InvalidURI': (400, 'Could not parse the specified URI')}
        resp = webob.Response(content_type='text/xml')
        resp.status = error_table[code][0]
        error_msg = '<?xml version="1.0" encoding="UTF-8"?>\r\n<Error>\r\n  <Code>%s</Code>\r\n  <Message>%s</Message>\r\n</Error>\r\n' % (code, error_table[code][1])
        error_msg = error_msg.encode()
        resp.body = error_msg
        return resp

    def _json_request(self, creds_json):
        headers = {'Content-Type': 'application/json'}
        try:
            response = requests.post('%s/v2.0/s3tokens' % self._request_uri, headers=headers, data=creds_json, verify=self._verify, timeout=CONF.s3_token.timeout)
        except requests.exceptions.RequestException as e:
            self._logger.info('HTTP connection exception: %s', e)
            resp = self._deny_request('InvalidURI')
            raise ServiceError(resp)
        if response.status_code < 200 or response.status_code >= 300:
            self._logger.debug('Keystone reply error: status=%s reason=%s', response.status_code, response.reason)
            resp = self._deny_request('AccessDenied')
            raise ServiceError(resp)
        return response

    def __call__(self, environ, start_response):
        """Handle incoming request. authenticate and send downstream."""
        req = webob.Request(environ)
        self._logger.debug('Calling S3Token middleware.')
        try:
            parts = strutils.split_path(req.path, 1, 4, True)
            version, account, container, obj = parts
        except ValueError:
            msg = 'Not a path query, skipping.'
            self._logger.debug(msg)
            return self._app(environ, start_response)
        if 'Authorization' not in req.headers:
            msg = 'No Authorization header. skipping.'
            self._logger.debug(msg)
            return self._app(environ, start_response)
        token = req.headers.get('X-Auth-Token', req.headers.get('X-Storage-Token'))
        if not token:
            msg = 'You did not specify an auth or a storage token. skipping.'
            self._logger.debug(msg)
            return self._app(environ, start_response)
        auth_header = req.headers['Authorization']
        try:
            access, signature = auth_header.split(' ')[-1].rsplit(':', 1)
        except ValueError:
            msg = 'You have an invalid Authorization header: %s'
            self._logger.debug(msg, auth_header)
            return self._deny_request('InvalidURI')(environ, start_response)
        force_tenant = None
        if ':' in access:
            access, force_tenant = access.split(':')
        creds = {'credentials': {'access': access, 'token': token, 'signature': signature}}
        creds_json = jsonutils.dumps(creds)
        self._logger.debug('Connecting to Keystone sending this JSON: %s', creds_json)
        try:
            resp = self._json_request(creds_json)
        except ServiceError as e:
            resp = e.args[0]
            msg = 'Received error, exiting middleware with error: %s'
            self._logger.debug(msg, resp.status_code)
            return resp(environ, start_response)
        self._logger.debug('Keystone Reply: Status: %d, Output: %s', resp.status_code, resp.content)
        try:
            identity_info = resp.json()
            token_id = str(identity_info['access']['token']['id'])
            tenant = identity_info['access']['token']['tenant']
        except (ValueError, KeyError):
            error = 'Error on keystone reply: %d %s'
            self._logger.debug(error, resp.status_code, resp.content)
            return self._deny_request('InvalidURI')(environ, start_response)
        req.headers['X-Auth-Token'] = token_id
        tenant_to_connect = force_tenant or tenant['id']
        self._logger.debug('Connecting with tenant: %s', tenant_to_connect)
        new_tenant_name = '%s%s' % (self._reseller_prefix, tenant_to_connect)
        environ['PATH_INFO'] = environ['PATH_INFO'].replace(account, new_tenant_name)
        return self._app(environ, start_response)