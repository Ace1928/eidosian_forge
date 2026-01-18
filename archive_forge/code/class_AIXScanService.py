from __future__ import absolute_import, division, print_function
import os
import platform
import re
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.locale import get_best_parsable_locale
class AIXScanService(BaseService):

    def gather_services(self):
        services = {}
        if platform.system() == 'AIX':
            lssrc_path = self.module.get_bin_path('lssrc')
            if lssrc_path:
                rc, stdout, stderr = self.module.run_command('%s -a' % lssrc_path)
                if rc != 0:
                    self.module.warn('lssrc could not retrieve service data (%s): %s' % (rc, stderr))
                else:
                    for line in stdout.split('\n'):
                        line_data = line.split()
                        if len(line_data) < 2:
                            continue
                        if line_data[0] == 'Subsystem':
                            continue
                        service_name = line_data[0]
                        if line_data[-1] == 'active':
                            service_state = 'running'
                        elif line_data[-1] == 'inoperative':
                            service_state = 'stopped'
                        else:
                            service_state = 'unknown'
                        services[service_name] = {'name': service_name, 'state': service_state, 'source': 'src'}
        return services