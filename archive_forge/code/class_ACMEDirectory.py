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
class ACMEDirectory(object):
    """
    The ACME server directory. Gives access to the available resources,
    and allows to obtain a Replay-Nonce. The acme_directory URL
    needs to support unauthenticated GET requests; ACME endpoints
    requiring authentication are not supported.
    https://tools.ietf.org/html/rfc8555#section-7.1.1
    """

    def __init__(self, module, account):
        self.module = module
        self.directory_root = module.params['acme_directory']
        self.version = module.params['acme_version']
        self.directory, dummy = account.get_request(self.directory_root, get_only=True)
        self.request_timeout = module.params['request_timeout']
        if self.version == 1:
            for key in ('new-reg', 'new-authz', 'new-cert'):
                if key not in self.directory:
                    raise ModuleFailException('ACME directory does not seem to follow protocol ACME v1')
        if self.version == 2:
            for key in ('newNonce', 'newAccount', 'newOrder'):
                if key not in self.directory:
                    raise ModuleFailException('ACME directory does not seem to follow protocol ACME v2')
            if 'meta' not in self.directory:
                self.directory['meta'] = {}

    def __getitem__(self, key):
        return self.directory[key]

    def get_nonce(self, resource=None):
        url = self.directory_root if self.version == 1 else self.directory['newNonce']
        if resource is not None:
            url = resource
        retry_count = 0
        while True:
            response, info = fetch_url(self.module, url, method='HEAD', timeout=self.request_timeout)
            if _decode_retry(self.module, response, info, retry_count):
                retry_count += 1
                continue
            if info['status'] not in (200, 204):
                raise NetworkException('Failed to get replay-nonce, got status {0}'.format(format_http_status(info['status'])))
            if 'replay-nonce' in info:
                return info['replay-nonce']
            self.module.log('HEAD to {0} did return status {1}, but no replay-nonce header!'.format(url, format_http_status(info['status'])))
            if retry_count >= 5:
                raise ACMEProtocolException(self.module, msg='Was not able to obtain nonce, giving up after 5 retries', info=info, response=response)
            retry_count += 1