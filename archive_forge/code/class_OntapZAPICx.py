from __future__ import (absolute_import, division, print_function)
import base64
import logging
import os
import ssl
import time
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils._text import to_native
class OntapZAPICx(zapi.NaServer):
    """ override zapi NaServer class to:
        - enable SSL certificate authentication
        - ignore invalid XML characters in ONTAP output (when using CLI module)
        - add Authorization header when using basic authentication
        """

    def __init__(self, hostname=None, server_type=zapi.NaServer.SERVER_TYPE_FILER, transport_type=zapi.NaServer.TRANSPORT_TYPE_HTTP, style=zapi.NaServer.STYLE_LOGIN_PASSWORD, username=None, password=None, port=None, trace=False, module=None, cert_filepath=None, key_filepath=None, validate_certs=None, auth_method=None):
        super(OntapZAPICx, self).__init__(hostname, server_type=server_type, transport_type=transport_type, style=style, username=username, password=password, port=port, trace=trace)
        self.cert_filepath = cert_filepath
        self.key_filepath = key_filepath
        self.validate_certs = validate_certs
        self.module = module
        self.base64_creds = None
        if auth_method == 'speedy_basic_auth':
            auth = '%s:%s' % (username, password)
            self.base64_creds = base64.b64encode(auth.encode()).decode()

    def _create_certificate_auth_handler(self):
        try:
            context = ssl.create_default_context()
        except AttributeError as exc:
            self._fail_with_exc_info('SSL certificate authentication requires python 2.7 or later.', exc)
        if not self.validate_certs:
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
        try:
            context.load_cert_chain(self.cert_filepath, keyfile=self.key_filepath)
        except IOError as exc:
            self._fail_with_exc_info('Cannot load SSL certificate, check files exist.', exc)
        return zapi.urllib.request.HTTPSHandler(context=context)

    def _fail_with_exc_info(self, arg0, exc):
        msg = arg0
        msg += '  More info: %s' % repr(exc)
        self.module.fail_json(msg=msg)

    def sanitize_xml(self, response):
        new_response = response.replace(b'\x07\n', b'')
        new_response = new_response.replace(b'\x07\r\n', b'')
        for code_point in get_feature(self.module, 'sanitize_code_points'):
            if bytes([8]) == b'\x08':
                byte = bytes([code_point])
            elif chr(8) == b'\x08':
                byte = chr(code_point)
            else:
                byte = b'.'
            new_response = new_response.replace(byte, b'.')
        return new_response

    def _parse_response(self, response):
        """ handling XML parsing exception """
        try:
            return super(OntapZAPICx, self)._parse_response(response)
        except zapi.etree.XMLSyntaxError as exc:
            if has_feature(self.module, 'sanitize_xml'):
                try:
                    return super(OntapZAPICx, self)._parse_response(self.sanitize_xml(response))
                except Exception:
                    pass
            try:
                exc.msg += '.  Received: %s' % response
            except Exception:
                pass
            raise exc

    def _create_request(self, na_element, enable_tunneling=False):
        """ intercept newly created request to add Authorization header """
        request, netapp_element = super(OntapZAPICx, self)._create_request(na_element, enable_tunneling=enable_tunneling)
        request.add_header('X-Dot-Client-App', CLIENT_APP_VERSION % self.module._name)
        if self.base64_creds is not None:
            request.add_header('Authorization', 'Basic %s' % self.base64_creds)
        return (request, netapp_element)

    def invoke_elem(self, na_element, enable_tunneling=False):
        """Invoke the API on the server."""
        if not na_element or not isinstance(na_element, zapi.NaElement):
            raise ValueError('NaElement must be supplied to invoke API')
        request, request_element = self._create_request(na_element, enable_tunneling)
        if self._trace:
            zapi.LOG.debug('Request: %s', request_element.to_string(pretty=True))
        if not hasattr(self, '_opener') or not self._opener or self._refresh_conn:
            self._build_opener()
        try:
            if hasattr(self, '_timeout'):
                response = self._opener.open(request, timeout=self._timeout)
            else:
                response = self._opener.open(request)
        except zapi.urllib.error.HTTPError as exc:
            raise zapi.NaApiError(exc.code, exc.reason)
        except zapi.urllib.error.URLError as exc:
            msg = 'URL error'
            error = repr(exc)
            try:
                if isinstance(exc.reason, ConnectionRefusedError):
                    msg = 'Unable to connect'
                    error = exc.args
            except Exception:
                pass
            raise zapi.NaApiError(msg, error)
        except Exception as exc:
            raise zapi.NaApiError('Unexpected error', repr(exc))
        response_xml = response.read()
        response_element = self._get_result(response_xml)
        if self._trace:
            zapi.LOG.debug('Response: %s', response_element.to_string(pretty=True))
        return response_element