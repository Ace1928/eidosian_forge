from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.api import basic_auth_argument_spec
from ansible_collections.community.general.plugins.module_utils.gitlab import (
class GitlabProjectVariables(object):

    def __init__(self, module, gitlab_instance):
        self.repo = gitlab_instance
        self.project = self.get_project(module.params['project'])
        self._module = module

    def get_project(self, project_name):
        return self.repo.projects.get(project_name)

    def list_all_project_variables(self):
        return list(self.project.variables.list(**list_all_kwargs))

    def create_variable(self, var_obj):
        if self._module.check_mode:
            return True
        var = {'key': var_obj.get('key'), 'value': var_obj.get('value'), 'masked': var_obj.get('masked'), 'protected': var_obj.get('protected'), 'raw': var_obj.get('raw'), 'variable_type': var_obj.get('variable_type')}
        if var_obj.get('environment_scope') is not None:
            var['environment_scope'] = var_obj.get('environment_scope')
        self.project.variables.create(var)
        return True

    def update_variable(self, var_obj):
        if self._module.check_mode:
            return True
        self.delete_variable(var_obj)
        self.create_variable(var_obj)
        return True

    def delete_variable(self, var_obj):
        if self._module.check_mode:
            return True
        self.project.variables.delete(var_obj.get('key'), filter={'environment_scope': var_obj.get('environment_scope')})
        return True