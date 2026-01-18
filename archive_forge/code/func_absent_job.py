from __future__ import absolute_import, division, print_function
import os
import traceback
import xml.etree.ElementTree as ET
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_native
def absent_job(self):
    if self.job_exists():
        self.result['changed'] = True
        self.result['diff']['before'] = self.get_current_config()
        if not self.module.check_mode:
            try:
                self.server.delete_job(self.name)
            except Exception as e:
                self.module.fail_json(msg='Unable to delete job, %s for %s' % (to_native(e), self.jenkins_url), exception=traceback.format_exc())