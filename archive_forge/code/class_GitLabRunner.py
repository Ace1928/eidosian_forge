from __future__ import absolute_import, division, print_function
from ansible.module_utils.api import basic_auth_argument_spec
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.general.plugins.module_utils.gitlab import (
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
class GitLabRunner(object):

    def __init__(self, module, gitlab_instance, group=None, project=None):
        self._module = module
        self._gitlab = gitlab_instance
        self.runner_object = None
        if project:
            self._runners_endpoint = project.runners.list
        elif group:
            self._runners_endpoint = group.runners.list
        elif module.params['owned']:
            self._runners_endpoint = gitlab_instance.runners.list
        else:
            self._runners_endpoint = gitlab_instance.runners.all

    def create_or_update_runner(self, description, options):
        changed = False
        arguments = {'locked': options['locked'], 'run_untagged': options['run_untagged'], 'maximum_timeout': options['maximum_timeout'], 'tag_list': options['tag_list']}
        if options.get('paused') is not None:
            arguments['paused'] = options['paused']
        else:
            arguments['active'] = options['active']
        if options.get('access_level') is not None:
            arguments['access_level'] = options['access_level']
        if self.runner_object is None:
            arguments['description'] = description
            if options.get('registration_token') is not None:
                arguments['token'] = options['registration_token']
            elif options.get('group') is not None:
                arguments['runner_type'] = 'group_type'
                arguments['group_id'] = options['group']
            elif options.get('project') is not None:
                arguments['runner_type'] = 'project_type'
                arguments['project_id'] = options['project']
            else:
                arguments['runner_type'] = 'instance_type'
            access_level_on_creation = self._module.params['access_level_on_creation']
            if not access_level_on_creation:
                arguments.pop('access_level', None)
            runner = self.create_runner(arguments)
            changed = True
        else:
            changed, runner = self.update_runner(self.runner_object, arguments)
            if changed:
                if self._module.check_mode:
                    self._module.exit_json(changed=True, msg='Successfully updated the runner %s' % description)
                try:
                    runner.save()
                except Exception as e:
                    self._module.fail_json(msg='Failed to update runner: %s ' % to_native(e))
        self.runner_object = runner
        return changed
    '\n    @param arguments Attributes of the runner\n    '

    def create_runner(self, arguments):
        if self._module.check_mode:
            return True
        try:
            if arguments.get('token') is not None:
                runner = self._gitlab.runners.create(arguments)
            elif LooseVersion(gitlab.__version__) < LooseVersion('4.0.0'):
                self._module.fail_json(msg='New runner creation workflow requires python-gitlab 4.0.0 or higher')
            else:
                runner = self._gitlab.user.runners.create(arguments)
        except gitlab.exceptions.GitlabCreateError as e:
            self._module.fail_json(msg='Failed to create runner: %s ' % to_native(e))
        return runner
    '\n    @param runner Runner object\n    @param arguments Attributes of the runner\n    '

    def update_runner(self, runner, arguments):
        changed = False
        for arg_key, arg_value in arguments.items():
            if arguments[arg_key] is not None:
                if isinstance(arguments[arg_key], list):
                    list1 = getattr(runner, arg_key)
                    list1.sort()
                    list2 = arguments[arg_key]
                    list2.sort()
                    if list1 != list2:
                        setattr(runner, arg_key, arguments[arg_key])
                        changed = True
                elif getattr(runner, arg_key) != arguments[arg_key]:
                    setattr(runner, arg_key, arguments[arg_key])
                    changed = True
        return (changed, runner)
    '\n    @param description Description of the runner\n    '

    def find_runner(self, description):
        runners = self._runners_endpoint(**list_all_kwargs)
        for runner in runners:
            if hasattr(runner, 'description'):
                if runner.description == description:
                    return self._gitlab.runners.get(runner.id)
            elif runner['description'] == description:
                return self._gitlab.runners.get(runner['id'])
    '\n    @param description Description of the runner\n    '

    def exists_runner(self, description):
        runner = self.find_runner(description)
        if runner:
            self.runner_object = runner
            return True
        return False

    def delete_runner(self):
        if self._module.check_mode:
            return True
        runner = self.runner_object
        return runner.delete()