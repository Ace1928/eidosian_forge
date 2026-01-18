from __future__ import absolute_import, division, print_function
import json
import socket
import getpass
from datetime import datetime
from ansible.module_utils._text import to_text
from ansible.module_utils.urls import open_url
from ansible.plugins.callback import CallbackBase
def _send_annotations(self, data):
    if self.dashboard_id:
        data['dashboardId'] = int(self.dashboard_id)
    if self.panel_ids:
        for panel_id in self.panel_ids:
            data['panelId'] = int(panel_id)
            self._send_annotation(data)
    else:
        self._send_annotation(data)