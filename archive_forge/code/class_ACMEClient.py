from __future__ import absolute_import, division, print_function
import copy
import datetime
import json
import locale
import time
import traceback
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.common.text.converters import to_bytes
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.six import PY3
from ansible_collections.community.crypto.plugins.module_utils.acme.backend_openssl_cli import (
from ansible_collections.community.crypto.plugins.module_utils.acme.backend_cryptography import (
from ansible_collections.community.crypto.plugins.module_utils.acme.errors import (
from ansible_collections.community.crypto.plugins.module_utils.acme.utils import (
class ACMEClient(object):
    """
    ACME client object. Handles the authorized communication with the
    ACME server.
    """

    def __init__(self, module, backend):
        self._debug = False
        self.module = module
        self.backend = backend
        self.version = module.params['acme_version']
        self.account_key_file = module.params['account_key_src']
        self.account_key_content = module.params['account_key_content']
        self.account_key_passphrase = module.params['account_key_passphrase']
        self.account_uri = module.params.get('account_uri') or None
        self.request_timeout = module.params['request_timeout']
        self.account_key_data = None
        self.account_jwk = None
        self.account_jws_header = None
        if self.account_key_file is not None or self.account_key_content is not None:
            try:
                self.account_key_data = self.parse_key(key_file=self.account_key_file, key_content=self.account_key_content, passphrase=self.account_key_passphrase)
            except KeyParsingError as e:
                raise ModuleFailException('Error while parsing account key: {msg}'.format(msg=e.msg))
            self.account_jwk = self.account_key_data['jwk']
            self.account_jws_header = {'alg': self.account_key_data['alg'], 'jwk': self.account_jwk}
            if self.account_uri:
                self.set_account_uri(self.account_uri)
        self.directory = ACMEDirectory(module, self)

    def set_account_uri(self, uri):
        """
        Set account URI. For ACME v2, it needs to be used to sending signed
        requests.
        """
        self.account_uri = uri
        if self.version != 1:
            self.account_jws_header.pop('jwk')
            self.account_jws_header['kid'] = self.account_uri

    def parse_key(self, key_file=None, key_content=None, passphrase=None):
        """
        Parses an RSA or Elliptic Curve key file in PEM format and returns key_data.
        In case of an error, raises KeyParsingError.
        """
        if key_file is None and key_content is None:
            raise AssertionError('One of key_file and key_content must be specified!')
        return self.backend.parse_key(key_file, key_content, passphrase=passphrase)

    def sign_request(self, protected, payload, key_data, encode_payload=True):
        """
        Signs an ACME request.
        """
        try:
            if payload is None:
                payload64 = ''
            else:
                if encode_payload:
                    payload = self.module.jsonify(payload).encode('utf8')
                payload64 = nopad_b64(to_bytes(payload))
            protected64 = nopad_b64(self.module.jsonify(protected).encode('utf8'))
        except Exception as e:
            raise ModuleFailException('Failed to encode payload / headers as JSON: {0}'.format(e))
        return self.backend.sign(payload64, protected64, key_data)

    def _log(self, msg, data=None):
        """
        Write arguments to acme.log when logging is enabled.
        """
        if self._debug:
            with open('acme.log', 'ab') as f:
                f.write('[{0}] {1}\n'.format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%s'), msg).encode('utf-8'))
                if data is not None:
                    f.write('{0}\n\n'.format(json.dumps(data, indent=2, sort_keys=True)).encode('utf-8'))

    def send_signed_request(self, url, payload, key_data=None, jws_header=None, parse_json_result=True, encode_payload=True, fail_on_error=True, error_msg=None, expected_status_codes=None):
        """
        Sends a JWS signed HTTP POST request to the ACME server and returns
        the response as dictionary (if parse_json_result is True) or in raw form
        (if parse_json_result is False).
        https://tools.ietf.org/html/rfc8555#section-6.2

        If payload is None, a POST-as-GET is performed.
        (https://tools.ietf.org/html/rfc8555#section-6.3)
        """
        key_data = key_data or self.account_key_data
        jws_header = jws_header or self.account_jws_header
        failed_tries = 0
        while True:
            protected = copy.deepcopy(jws_header)
            protected['nonce'] = self.directory.get_nonce()
            if self.version != 1:
                protected['url'] = url
            self._log('URL', url)
            self._log('protected', protected)
            self._log('payload', payload)
            data = self.sign_request(protected, payload, key_data, encode_payload=encode_payload)
            if self.version == 1:
                data['header'] = jws_header.copy()
                for k, v in protected.items():
                    dummy = data['header'].pop(k, None)
            self._log('signed request', data)
            data = self.module.jsonify(data)
            headers = {'Content-Type': 'application/jose+json'}
            resp, info = fetch_url(self.module, url, data=data, headers=headers, method='POST', timeout=self.request_timeout)
            if _decode_retry(self.module, resp, info, failed_tries):
                failed_tries += 1
                continue
            _assert_fetch_url_success(self.module, resp, info)
            result = {}
            try:
                if PY3 and resp.closed:
                    raise TypeError
                content = resp.read()
            except (AttributeError, TypeError):
                content = info.pop('body', None)
            if content or not parse_json_result:
                if parse_json_result and info['content-type'].startswith('application/json') or 400 <= info['status'] < 600:
                    try:
                        decoded_result = self.module.from_json(content.decode('utf8'))
                        self._log('parsed result', decoded_result)
                        if all((400 <= info['status'] < 600, decoded_result.get('type') == 'urn:ietf:params:acme:error:badNonce', failed_tries <= 5)):
                            failed_tries += 1
                            continue
                        if parse_json_result:
                            result = decoded_result
                        else:
                            result = content
                    except ValueError:
                        raise NetworkException('Failed to parse the ACME response: {0} {1}'.format(url, content))
                else:
                    result = content
            if fail_on_error and _is_failed(info, expected_status_codes=expected_status_codes):
                raise ACMEProtocolException(self.module, msg=error_msg, info=info, content=content, content_json=result if parse_json_result else None)
            return (result, info)

    def get_request(self, uri, parse_json_result=True, headers=None, get_only=False, fail_on_error=True, error_msg=None, expected_status_codes=None):
        """
        Perform a GET-like request. Will try POST-as-GET for ACMEv2, with fallback
        to GET if server replies with a status code of 405.
        """
        if not get_only and self.version != 1:
            content, info = self.send_signed_request(uri, None, parse_json_result=False, fail_on_error=False)
            if info['status'] == 405:
                get_only = True
        else:
            get_only = True
        if get_only:
            retry_count = 0
            while True:
                resp, info = fetch_url(self.module, uri, method='GET', headers=headers, timeout=self.request_timeout)
                if not _decode_retry(self.module, resp, info, retry_count):
                    break
                retry_count += 1
            _assert_fetch_url_success(self.module, resp, info)
            try:
                if PY3 and resp.closed:
                    raise TypeError
                content = resp.read()
            except (AttributeError, TypeError):
                content = info.pop('body', None)
        parsed_json_result = False
        if parse_json_result:
            result = {}
            if content:
                if info['content-type'].startswith('application/json'):
                    try:
                        result = self.module.from_json(content.decode('utf8'))
                        parsed_json_result = True
                    except ValueError:
                        raise NetworkException('Failed to parse the ACME response: {0} {1}'.format(uri, content))
                else:
                    result = content
        else:
            result = content
        if fail_on_error and _is_failed(info, expected_status_codes=expected_status_codes):
            raise ACMEProtocolException(self.module, msg=error_msg, info=info, content=content, content_json=result if parsed_json_result else None)
        return (result, info)