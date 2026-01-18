from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import _load_params
import datetime
from ansible.module_utils.six import raise_from
def _generate_playbook(self, counter, export_path, selector, robject, state_present, need_bypass, url_params, params_schema, log):
    prefix_text = '- name: Exported Playbook\n  hosts: fortimanagers\n  connection: httpapi\n  collections:\n    - fortinet.fortimanager\n  vars:\n    ansible_httpapi_use_ssl: true\n    ansible_httpapi_validate_certs: false\n    ansible_httpapi_port: 443\n  tasks:\n'
    with open('%s/%s_%s.yml' % (export_path, selector, counter), 'w') as f:
        f.write(prefix_text)
        f.write('  - name: exported config for %s\n' % selector)
        f.write('    fmgr_%s:\n' % selector)
        if need_bypass:
            f.write('      bypass_validation: true\n')
        if state_present:
            f.write('      state: present\n')
        for url_param_key in params_schema:
            if url_param_key not in url_params:
                continue
            url_param_value = url_params[url_param_key]
            f.write('      %s: %s\n' % (url_param_key, url_param_value))
        f.write('      %s:\n' % selector)
        f.write(self.__append_whiteblank_per_line(yaml.dump(robject), 8))
    log.write('\texported playbook: %s/%s_%s.yml\n' % (export_path, selector, counter))
    self._nr_exported_playbooks += 1