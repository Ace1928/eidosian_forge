from __future__ import absolute_import, division, print_function
import re
from ansible_collections.community.general.plugins.module_utils.module_helper import StateModuleHelper
from ansible_collections.community.general.plugins.module_utils.snap import snap_runner
class SnapAlias(StateModuleHelper):
    _RE_ALIAS_LIST = re.compile('^(?P<snap>\\S+)\\s+(?P<alias>[\\w-]+)\\s+.*$')
    module = dict(argument_spec={'state': dict(type='str', choices=['absent', 'present'], default='present'), 'name': dict(type='str'), 'alias': dict(type='list', elements='str', aliases=['aliases'])}, required_if=[('state', 'present', ['name', 'alias']), ('state', 'absent', ['name', 'alias'], True)], supports_check_mode=True)

    def _aliases(self):
        n = self.vars.name
        return {n: self._get_aliases_for(n)} if n else self._get_aliases()

    def __init_module__(self):
        self.runner = snap_runner(self.module)
        self.vars.set('snap_aliases', self._aliases(), change=True, diff=True)

    def __quit_module__(self):
        self.vars.snap_aliases = self._aliases()

    def _get_aliases(self):

        def process(rc, out, err):
            if err:
                return {}
            aliases = [self._RE_ALIAS_LIST.match(a.strip()) for a in out.splitlines()[1:]]
            snap_alias_list = [(entry.group('snap'), entry.group('alias')) for entry in aliases]
            results = {}
            for snap, alias in snap_alias_list:
                results[snap] = results.get(snap, []) + [alias]
            return results
        with self.runner('state_alias name', check_rc=True, output_process=process) as ctx:
            aliases = ctx.run(state_alias='info')
            if self.verbosity >= 4:
                self.vars.get_aliases_run_info = ctx.run_info
            return aliases

    def _get_aliases_for(self, name):
        return self._get_aliases().get(name, [])

    def _has_alias(self, name=None, alias=None):
        if name:
            if name not in self.vars.snap_aliases:
                return False
            if alias is None:
                return bool(self.vars.snap_aliases[name])
            return alias in self.vars.snap_aliases[name]
        return any((alias in aliases for aliases in self.vars.snap_aliases.values()))

    def state_present(self):
        for _alias in self.vars.alias:
            if not self._has_alias(self.vars.name, _alias):
                self.changed = True
                with self.runner('state_alias name alias', check_mode_skip=True) as ctx:
                    ctx.run(state_alias=self.vars.state, alias=_alias)
                    if self.verbosity >= 4:
                        self.vars.run_info = ctx.run_info

    def state_absent(self):
        if not self.vars.alias:
            if self._has_alias(self.vars.name):
                self.changed = True
                with self.runner('state_alias name', check_mode_skip=True) as ctx:
                    ctx.run(state_alias=self.vars.state)
                    if self.verbosity >= 4:
                        self.vars.run_info = ctx.run_info
        else:
            for _alias in self.vars.alias:
                if self._has_alias(self.vars.name, _alias):
                    self.changed = True
                    with self.runner('state_alias alias', check_mode_skip=True) as ctx:
                        ctx.run(state_alias=self.vars.state, alias=_alias)
                        if self.verbosity >= 4:
                            self.vars.run_info = ctx.run_info