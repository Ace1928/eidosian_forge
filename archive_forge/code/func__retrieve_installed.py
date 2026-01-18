from __future__ import absolute_import, division, print_function
import json
from ansible_collections.community.general.plugins.module_utils.module_helper import StateModuleHelper
from ansible_collections.community.general.plugins.module_utils.pipx import pipx_runner
from ansible.module_utils.facts.compat import ansible_facts
def _retrieve_installed(self):

    def process_list(rc, out, err):
        if not out:
            return {}
        results = {}
        raw_data = json.loads(out)
        for venv_name, venv in raw_data['venvs'].items():
            results[venv_name] = {'version': venv['metadata']['main_package']['package_version'], 'injected': dict(((k, v['package_version']) for k, v in venv['metadata']['injected_packages'].items()))}
        return results
    installed = self.runner('_list', output_process=process_list).run(_list=1)
    if self.vars.name is not None:
        app_list = installed.get(self.vars.name)
        if app_list:
            return {self.vars.name: app_list}
        else:
            return {}
    return installed