import contextlib
import copy
import os
from oslo_policy import _checks
from oslo_policy._i18n import _
from oslo_serialization import jsonutils
import requests
class HttpsCheck(HttpCheck):
    """Check ``https:`` rules by calling to a remote server.

    This example implementation simply verifies that the response
    is exactly ``True``.
    """

    def __call__(self, target, creds, enforcer, current_rule=None):
        url = ('https:' + self.match) % target
        cert_file = enforcer.conf.oslo_policy.remote_ssl_client_crt_file
        key_file = enforcer.conf.oslo_policy.remote_ssl_client_key_file
        ca_crt_file = enforcer.conf.oslo_policy.remote_ssl_ca_crt_file
        verify_server = enforcer.conf.oslo_policy.remote_ssl_verify_server_crt
        if cert_file:
            if not os.path.exists(cert_file):
                raise RuntimeError(_('Unable to find ssl cert_file  : %s') % cert_file)
            if not os.access(cert_file, os.R_OK):
                raise RuntimeError(_('Unable to access ssl cert_file  : %s') % cert_file)
        if key_file:
            if not os.path.exists(key_file):
                raise RuntimeError(_('Unable to find ssl key_file : %s') % key_file)
            if not os.access(key_file, os.R_OK):
                raise RuntimeError(_('Unable to access ssl key_file  : %s') % key_file)
        cert = (cert_file, key_file)
        if verify_server:
            if ca_crt_file:
                if not os.path.exists(ca_crt_file):
                    raise RuntimeError(_('Unable to find ca cert_file  : %s') % ca_crt_file)
                verify_server = ca_crt_file
        data, json = self._construct_payload(creds, current_rule, enforcer, target)
        with contextlib.closing(requests.post(url, json=json, data=data, cert=cert, verify=verify_server)) as r:
            return r.text.lstrip('"').rstrip('"') == 'True'