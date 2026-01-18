from __future__ import absolute_import, division, print_function
import hashlib
import io
import json
import os
import tempfile
from ansible.module_utils.basic import AnsibleModule, to_bytes
from ansible.module_utils.six.moves import http_cookiejar as cookiejar
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible.module_utils.urls import fetch_url, url_argument_spec
from ansible.module_utils.six import text_type, binary_type
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.general.plugins.module_utils.jenkins import download_updates_file
def _get_urls_data(self, urls, what=None, msg_status=None, msg_exception=None, **kwargs):
    if msg_status is None:
        msg_status = 'Cannot get %s' % what
    if msg_exception is None:
        msg_exception = 'Retrieval of %s failed.' % what
    errors = {}
    for url in urls:
        err_msg = None
        try:
            self.module.debug('fetching url: %s' % url)
            response, info = fetch_url(self.module, url, timeout=self.timeout, cookies=self.cookies, headers=self.crumb, **kwargs)
            if info['status'] == 200:
                return response
            else:
                err_msg = '%s. fetching url %s failed. response code: %s' % (msg_status, url, info['status'])
                if info['status'] > 400:
                    err_msg = '%s. response body: %s' % (err_msg, info['body'])
        except Exception as e:
            err_msg = '%s. fetching url %s failed. error msg: %s' % (msg_status, url, to_native(e))
        finally:
            if err_msg is not None:
                self.module.debug(err_msg)
                errors[url] = err_msg
    self.module.fail_json(msg=msg_exception, details=errors)