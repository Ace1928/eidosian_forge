from __future__ import absolute_import, division, print_function
import json
from ansible_collections.community.general.plugins.module_utils.module_helper import ModuleHelper
from ansible_collections.community.general.plugins.module_utils.pipx import pipx_runner
from ansible.module_utils.facts.compat import ansible_facts
class PipXInfo(ModuleHelper):
    output_params = ['name']
    module = dict(argument_spec=dict(name=dict(type='str'), include_deps=dict(type='bool', default=False), include_injected=dict(type='bool', default=False), include_raw=dict(type='bool', default=False), executable=dict(type='path')), supports_check_mode=True)

    def __init_module__(self):
        if self.vars.executable:
            self.command = [self.vars.executable]
        else:
            facts = ansible_facts(self.module, gather_subset=['python'])
            self.command = [facts['python']['executable'], '-m', 'pipx']
        self.runner = pipx_runner(self.module, self.command)

    def __run__(self):

        def process_list(rc, out, err):
            if not out:
                return []
            results = []
            raw_data = json.loads(out)
            if self.vars.include_raw:
                self.vars.raw_output = raw_data
            if self.vars.name:
                if self.vars.name in raw_data['venvs']:
                    data = {self.vars.name: raw_data['venvs'][self.vars.name]}
                else:
                    data = {}
            else:
                data = raw_data['venvs']
            for venv_name, venv in data.items():
                entry = {'name': venv_name, 'version': venv['metadata']['main_package']['package_version']}
                if self.vars.include_injected:
                    entry['injected'] = dict(((k, v['package_version']) for k, v in venv['metadata']['injected_packages'].items()))
                if self.vars.include_deps:
                    entry['dependencies'] = list(venv['metadata']['main_package']['app_paths_of_dependencies'])
                results.append(entry)
            return results
        with self.runner('_list', output_process=process_list) as ctx:
            self.vars.application = ctx.run(_list=1)
            self._capture_results(ctx)

    def _capture_results(self, ctx):
        self.vars.cmd = ctx.cmd
        if self.verbosity >= 4:
            self.vars.run_info = ctx.run_info