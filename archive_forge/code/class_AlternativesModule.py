from __future__ import absolute_import, division, print_function
import os
import re
from ansible.module_utils.basic import AnsibleModule
class AlternativesModule(object):
    _UPDATE_ALTERNATIVES = None

    def __init__(self, module):
        self.module = module
        self.result = dict(changed=False, diff=dict(before=dict(), after=dict()))
        self.module.run_command_environ_update = {'LC_ALL': 'C'}
        self.messages = []
        self.run()

    @property
    def mode_present(self):
        return self.module.params.get('state') in [AlternativeState.PRESENT, AlternativeState.SELECTED, AlternativeState.AUTO]

    @property
    def mode_selected(self):
        return self.module.params.get('state') == AlternativeState.SELECTED

    @property
    def mode_auto(self):
        return self.module.params.get('state') == AlternativeState.AUTO

    def run(self):
        self.parse()
        if self.mode_present:
            subcommands_parameter = self.module.params['subcommands']
            priority_parameter = self.module.params['priority']
            if self.path not in self.current_alternatives or (priority_parameter is not None and self.current_alternatives[self.path].get('priority') != priority_parameter) or (subcommands_parameter is not None and (not all((s in subcommands_parameter for s in self.current_alternatives[self.path].get('subcommands'))) or not all((s in self.current_alternatives[self.path].get('subcommands') for s in subcommands_parameter)))):
                self.install()
            if self.mode_selected and self.current_path != self.path:
                self.set()
            if self.mode_auto and self.current_mode == 'manual':
                self.auto()
        elif self.path in self.current_alternatives:
            self.remove()
        self.result['msg'] = ' '.join(self.messages)
        self.module.exit_json(**self.result)

    def install(self):
        if not os.path.exists(self.path):
            self.module.fail_json(msg='Specified path %s does not exist' % self.path)
        if not self.link:
            self.module.fail_json(msg='Needed to install the alternative, but unable to do so as we are missing the link')
        cmd = [self.UPDATE_ALTERNATIVES, '--install', self.link, self.name, self.path, str(self.priority)]
        if self.module.params['subcommands'] is not None:
            subcommands = [['--slave', subcmd['link'], subcmd['name'], subcmd['path']] for subcmd in self.subcommands]
            cmd += [item for sublist in subcommands for item in sublist]
        self.result['changed'] = True
        self.messages.append("Install alternative '%s' for '%s'." % (self.path, self.name))
        if not self.module.check_mode:
            self.module.run_command(cmd, check_rc=True)
        if self.module._diff:
            self.result['diff']['after'] = dict(state=AlternativeState.PRESENT, path=self.path, priority=self.priority, link=self.link)
            if self.subcommands:
                self.result['diff']['after'].update(dict(subcommands=self.subcommands))

    def remove(self):
        cmd = [self.UPDATE_ALTERNATIVES, '--remove', self.name, self.path]
        self.result['changed'] = True
        self.messages.append("Remove alternative '%s' from '%s'." % (self.path, self.name))
        if not self.module.check_mode:
            self.module.run_command(cmd, check_rc=True)
        if self.module._diff:
            self.result['diff']['after'] = dict(state=AlternativeState.ABSENT)

    def set(self):
        cmd = [self.UPDATE_ALTERNATIVES, '--set', self.name, self.path]
        self.result['changed'] = True
        self.messages.append("Set alternative '%s' for '%s'." % (self.path, self.name))
        if not self.module.check_mode:
            self.module.run_command(cmd, check_rc=True)
        if self.module._diff:
            self.result['diff']['after']['state'] = AlternativeState.SELECTED

    def auto(self):
        cmd = [self.UPDATE_ALTERNATIVES, '--auto', self.name]
        self.messages.append("Set alternative to auto for '%s'." % self.name)
        self.result['changed'] = True
        if not self.module.check_mode:
            self.module.run_command(cmd, check_rc=True)
        if self.module._diff:
            self.result['diff']['after']['state'] = AlternativeState.PRESENT

    @property
    def name(self):
        return self.module.params.get('name')

    @property
    def path(self):
        return self.module.params.get('path')

    @property
    def link(self):
        return self.module.params.get('link') or self.current_link

    @property
    def priority(self):
        if self.module.params.get('priority') is not None:
            return self.module.params.get('priority')
        return self.current_alternatives.get(self.path, {}).get('priority', 50)

    @property
    def subcommands(self):
        if self.module.params.get('subcommands') is not None:
            return self.module.params.get('subcommands')
        elif self.path in self.current_alternatives and self.current_alternatives[self.path].get('subcommands'):
            return self.current_alternatives[self.path].get('subcommands')
        return None

    @property
    def UPDATE_ALTERNATIVES(self):
        if self._UPDATE_ALTERNATIVES is None:
            self._UPDATE_ALTERNATIVES = self.module.get_bin_path('update-alternatives', True)
        return self._UPDATE_ALTERNATIVES

    def parse(self):
        self.current_mode = None
        self.current_path = None
        self.current_link = None
        self.current_alternatives = {}
        rc, display_output, dummy = self.module.run_command([self.UPDATE_ALTERNATIVES, '--display', self.name])
        if rc != 0:
            self.module.debug("No current alternative found. '%s' exited with %s" % (self.UPDATE_ALTERNATIVES, rc))
            return
        current_mode_regex = re.compile('\\s-\\s(?:status\\sis\\s)?(\\w*)(?:\\smode|.)$', re.MULTILINE)
        current_path_regex = re.compile('^\\s*link currently points to (.*)$', re.MULTILINE)
        current_link_regex = re.compile('^\\s*link \\w+ is (.*)$', re.MULTILINE)
        subcmd_path_link_regex = re.compile('^\\s*(?:slave|follower) (\\S+) is (.*)$', re.MULTILINE)
        alternative_regex = re.compile('^(\\/.*)\\s-\\s(?:family\\s\\S+\\s)?priority\\s(\\d+)((?:\\s+(?:slave|follower).*)*)', re.MULTILINE)
        subcmd_regex = re.compile('^\\s+(?:slave|follower) (.*): (.*)$', re.MULTILINE)
        match = current_mode_regex.search(display_output)
        if not match:
            self.module.debug('No current mode found in output')
            return
        self.current_mode = match.group(1)
        match = current_path_regex.search(display_output)
        if not match:
            self.module.debug('No current path found in output')
        else:
            self.current_path = match.group(1)
        match = current_link_regex.search(display_output)
        if not match:
            self.module.debug('No current link found in output')
        else:
            self.current_link = match.group(1)
        subcmd_path_map = dict(subcmd_path_link_regex.findall(display_output))
        if not subcmd_path_map and self.subcommands:
            subcmd_path_map = dict(((s['name'], s['link']) for s in self.subcommands))
        for path, prio, subcmd in alternative_regex.findall(display_output):
            self.current_alternatives[path] = dict(priority=int(prio), subcommands=[dict(name=name, path=spath, link=subcmd_path_map.get(name)) for name, spath in subcmd_regex.findall(subcmd) if spath != '(null)'])
        if self.module._diff:
            if self.path in self.current_alternatives:
                self.result['diff']['before'].update(dict(state=AlternativeState.PRESENT, path=self.path, priority=self.current_alternatives[self.path].get('priority'), link=self.current_link))
                if self.current_alternatives[self.path].get('subcommands'):
                    self.result['diff']['before'].update(dict(subcommands=self.current_alternatives[self.path].get('subcommands')))
                if self.current_mode == 'manual' and self.current_path != self.path:
                    self.result['diff']['before'].update(dict(state=AlternativeState.SELECTED))
            else:
                self.result['diff']['before'].update(dict(state=AlternativeState.ABSENT))