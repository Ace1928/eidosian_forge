from __future__ import absolute_import, division, print_function
import json
from ansible_collections.community.general.plugins.module_utils.module_helper import StateModuleHelper
from ansible_collections.community.general.plugins.module_utils.pipx import pipx_runner
from ansible.module_utils.facts.compat import ansible_facts
class PipX(StateModuleHelper):
    output_params = ['name', 'source', 'index_url', 'force', 'installdeps']
    module = dict(argument_spec=dict(state=dict(type='str', default='install', choices=['present', 'absent', 'install', 'uninstall', 'uninstall_all', 'inject', 'upgrade', 'upgrade_all', 'reinstall', 'reinstall_all', 'latest']), name=dict(type='str'), source=dict(type='str'), install_apps=dict(type='bool', default=False), install_deps=dict(type='bool', default=False), inject_packages=dict(type='list', elements='str'), force=dict(type='bool', default=False), include_injected=dict(type='bool', default=False), index_url=dict(type='str'), python=dict(type='str'), system_site_packages=dict(type='bool', default=False), executable=dict(type='path'), editable=dict(type='bool', default=False), pip_args=dict(type='str')), required_if=[('state', 'present', ['name']), ('state', 'install', ['name']), ('state', 'absent', ['name']), ('state', 'uninstall', ['name']), ('state', 'upgrade', ['name']), ('state', 'reinstall', ['name']), ('state', 'latest', ['name']), ('state', 'inject', ['name', 'inject_packages'])], supports_check_mode=True)

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

    def __init_module__(self):
        if self.vars.executable:
            self.command = [self.vars.executable]
        else:
            facts = ansible_facts(self.module, gather_subset=['python'])
            self.command = [facts['python']['executable'], '-m', 'pipx']
        self.runner = pipx_runner(self.module, self.command)
        self.vars.set('application', self._retrieve_installed(), change=True, diff=True)

    def __quit_module__(self):
        self.vars.application = self._retrieve_installed()

    def _capture_results(self, ctx):
        self.vars.stdout = ctx.results_out
        self.vars.stderr = ctx.results_err
        self.vars.cmd = ctx.cmd
        if self.verbosity >= 4:
            self.vars.run_info = ctx.run_info

    def state_install(self):
        if not self.vars.application or self.vars.force:
            self.changed = True
            with self.runner('state index_url install_deps force python system_site_packages editable pip_args name_source', check_mode_skip=True) as ctx:
                ctx.run(name_source=[self.vars.name, self.vars.source])
                self._capture_results(ctx)
    state_present = state_install

    def state_upgrade(self):
        if not self.vars.application:
            self.do_raise('Trying to upgrade a non-existent application: {0}'.format(self.vars.name))
        if self.vars.force:
            self.changed = True
        with self.runner('state include_injected index_url force editable pip_args name', check_mode_skip=True) as ctx:
            ctx.run()
            self._capture_results(ctx)

    def state_uninstall(self):
        if self.vars.application:
            with self.runner('state name', check_mode_skip=True) as ctx:
                ctx.run()
                self._capture_results(ctx)
    state_absent = state_uninstall

    def state_reinstall(self):
        if not self.vars.application:
            self.do_raise('Trying to reinstall a non-existent application: {0}'.format(self.vars.name))
        self.changed = True
        with self.runner('state name python', check_mode_skip=True) as ctx:
            ctx.run()
            self._capture_results(ctx)

    def state_inject(self):
        if not self.vars.application:
            self.do_raise('Trying to inject packages into a non-existent application: {0}'.format(self.vars.name))
        if self.vars.force:
            self.changed = True
        with self.runner('state index_url install_apps install_deps force editable pip_args name inject_packages', check_mode_skip=True) as ctx:
            ctx.run()
            self._capture_results(ctx)

    def state_uninstall_all(self):
        with self.runner('state', check_mode_skip=True) as ctx:
            ctx.run()
            self._capture_results(ctx)

    def state_reinstall_all(self):
        with self.runner('state python', check_mode_skip=True) as ctx:
            ctx.run()
            self._capture_results(ctx)

    def state_upgrade_all(self):
        if self.vars.force:
            self.changed = True
        with self.runner('state include_injected force', check_mode_skip=True) as ctx:
            ctx.run()
            self._capture_results(ctx)

    def state_latest(self):
        if not self.vars.application or self.vars.force:
            self.changed = True
            with self.runner('state index_url install_deps force python system_site_packages editable pip_args name_source', check_mode_skip=True) as ctx:
                ctx.run(state='install', name_source=[self.vars.name, self.vars.source])
                self._capture_results(ctx)
        with self.runner('state include_injected index_url force editable pip_args name', check_mode_skip=True) as ctx:
            ctx.run(state='upgrade')
            self._capture_results(ctx)