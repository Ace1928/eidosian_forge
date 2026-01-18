from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.api import basic_auth_argument_spec
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible_collections.community.general.plugins.module_utils.gitlab import (
class GitlabLabels(object):

    def __init__(self, module, gitlab_instance, group_id, project_id):
        self._gitlab = gitlab_instance
        self.gitlab_object = group_id if group_id else project_id
        self.is_group_label = True if group_id else False
        self._module = module

    def list_all_labels(self):
        page_nb = 1
        labels = []
        vars_page = self.gitlab_object.labels.list(page=page_nb)
        while len(vars_page) > 0:
            labels += vars_page
            page_nb += 1
            vars_page = self.gitlab_object.labels.list(page=page_nb)
        return labels

    def create_label(self, var_obj):
        if self._module.check_mode:
            return (True, True)
        var = {'name': var_obj.get('name'), 'color': var_obj.get('color')}
        if var_obj.get('description') is not None:
            var['description'] = var_obj.get('description')
        if var_obj.get('priority') is not None:
            var['priority'] = var_obj.get('priority')
        _obj = self.gitlab_object.labels.create(var)
        return (True, _obj.asdict())

    def update_label(self, var_obj):
        if self._module.check_mode:
            return (True, True)
        _label = self.gitlab_object.labels.get(var_obj.get('name'))
        if var_obj.get('new_name') is not None:
            _label.new_name = var_obj.get('new_name')
        if var_obj.get('description') is not None:
            _label.description = var_obj.get('description')
        if var_obj.get('priority') is not None:
            _label.priority = var_obj.get('priority')
        _label.save()
        return (True, _label.asdict())

    def delete_label(self, var_obj):
        if self._module.check_mode:
            return (True, True)
        _label = self.gitlab_object.labels.get(var_obj.get('name'))
        _label.delete()
        return (True, _label.asdict())