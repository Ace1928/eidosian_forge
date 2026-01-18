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
def _install_with_plugin_manager(self):
    if not self.module.check_mode:
        install_script = 'd = Jenkins.instance.updateCenter.getPlugin("%s").deploy(); d.get();' % self.params['name']
        if self.params['with_dependencies']:
            install_script = 'Jenkins.instance.updateCenter.getPlugin("%s").getNeededDependencies().each{it.deploy()}; %s' % (self.params['name'], install_script)
        script_data = {'script': install_script}
        data = urlencode(script_data)
        r = self._get_url_data('%s/scriptText' % self.url, msg_status='Cannot install plugin.', msg_exception='Plugin installation has failed.', data=data, dont_fail=True)
        hpi_file = '%s/plugins/%s.hpi' % (self.params['jenkins_home'], self.params['name'])
        if os.path.isfile(hpi_file):
            os.remove(hpi_file)