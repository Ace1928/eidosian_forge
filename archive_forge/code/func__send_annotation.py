from __future__ import absolute_import, division, print_function
import json
import socket
import getpass
from datetime import datetime
from ansible.module_utils._text import to_text
from ansible.module_utils.urls import open_url
from ansible.plugins.callback import CallbackBase
def _send_annotation(self, annotation):
    try:
        open_url(self.grafana_url, data=json.dumps(annotation), headers=self.headers, method='POST', validate_certs=self.validate_grafana_certs, url_username=self.grafana_user, url_password=self.grafana_password, http_agent=self.http_agent, force_basic_auth=self.force_basic_auth)
    except Exception as e:
        self._display.error('Could not submit message to Grafana: %s' % to_text(e))