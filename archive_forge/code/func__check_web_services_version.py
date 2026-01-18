from __future__ import absolute_import, division, print_function
import json
import random
import mimetypes
from pprint import pformat
from ansible.module_utils import six
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.six.moves.urllib.error import HTTPError, URLError
from ansible.module_utils.urls import open_url
from ansible.module_utils.api import basic_auth_argument_spec
from ansible.module_utils._text import to_native
def _check_web_services_version(self):
    """Verify proxy or embedded web services meets minimum version required for module.

        The minimum required web services version is evaluated against version supplied through the web services rest
        api. AnsibleFailJson exception will be raised when the minimum is not met or exceeded.

        This helper function will update the supplied api url if secure http is not used for embedded web services

        :raise AnsibleFailJson: raised when the contacted api service does not meet the minimum required version.
        """
    if not self.is_web_services_valid_cache:
        url_parts = urlparse(self.url)
        if not url_parts.scheme or not url_parts.netloc:
            self.module.fail_json(msg='Failed to provide valid API URL. Example: https://192.168.1.100:8443/devmgr/v2. URL [%s].' % self.url)
        if url_parts.scheme not in ['http', 'https']:
            self.module.fail_json(msg='Protocol must be http or https. URL [%s].' % self.url)
        self.url = '%s://%s/' % (url_parts.scheme, url_parts.netloc)
        about_url = self.url + self.DEFAULT_REST_API_ABOUT_PATH
        rc, data = request(about_url, timeout=self.DEFAULT_TIMEOUT, headers=self.DEFAULT_HEADERS, ignore_errors=True, force_basic_auth=False, **self.creds)
        if rc != 200:
            self.module.warn('Failed to retrieve web services about information! Retrying with secure ports. Array Id [%s].' % self.ssid)
            self.url = 'https://%s:8443/' % url_parts.netloc.split(':')[0]
            about_url = self.url + self.DEFAULT_REST_API_ABOUT_PATH
            try:
                rc, data = request(about_url, timeout=self.DEFAULT_TIMEOUT, headers=self.DEFAULT_HEADERS, **self.creds)
            except Exception as error:
                self.module.fail_json(msg='Failed to retrieve the webservices about information! Array Id [%s]. Error [%s].' % (self.ssid, to_native(error)))
        if len(data['version'].split('.')) == 4:
            major, minor, other, revision = data['version'].split('.')
            minimum_major, minimum_minor, other, minimum_revision = self.web_services_version.split('.')
            if not (major > minimum_major or (major == minimum_major and minor > minimum_minor) or (major == minimum_major and minor == minimum_minor and (revision >= minimum_revision))):
                self.module.fail_json(msg='Web services version does not meet minimum version required. Current version: [%s]. Version required: [%s].' % (data['version'], self.web_services_version))
            self.module.log('Web services rest api version met the minimum required version.')
        else:
            self.module.warn('Web services rest api version unknown!')
        self._check_ssid()
        self.is_web_services_valid_cache = True