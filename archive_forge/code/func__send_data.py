from __future__ import (absolute_import, division, print_function)
import os
from datetime import datetime
from collections import defaultdict
import json
import time
from ansible.module_utils._text import to_text
from ansible.module_utils.parsing.convert_bool import boolean as to_bool
from ansible.plugins.callback import CallbackBase
def _send_data(self, data_type, report_type, host, data):
    if data_type == 'facts':
        url = self.foreman_url + '/api/v2/hosts/facts'
    elif data_type == 'report' and report_type == 'foreman':
        url = self.foreman_url + '/api/v2/config_reports'
    elif data_type == 'report' and report_type == 'proxy':
        url = self.proxy_url + '/reports/ansible'
    else:
        self._display.warning(u'Unknown report_type: {rt}'.format(rt=report_type))
    if len(self.dir_store) > 0:
        filename = u'{host}.json'.format(host=to_text(host))
        filename = os.path.join(self.dir_store, filename)
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, sort_keys=True)
    else:
        try:
            response = self.session.post(url=url, json=data)
            response.raise_for_status()
        except requests.exceptions.RequestException as err:
            self._display.warning(u'Sending data to Foreman at {url} failed for {host}: {err}'.format(host=to_text(host), err=to_text(err), url=to_text(self.foreman_url)))