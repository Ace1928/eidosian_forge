from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.api import basic_auth_argument_spec
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible_collections.community.general.plugins.module_utils.gitlab import (
class GitlabMergeRequest(object):

    def __init__(self, module, project, gitlab_instance):
        self._gitlab = gitlab_instance
        self._module = module
        self.project = project
    '\n    @param branch Name of the branch\n    '

    def get_branch(self, branch):
        try:
            return self.project.branches.get(branch)
        except gitlab.exceptions.GitlabGetError as e:
            self._module.fail_json(msg='Failed to get the branch: %s' % to_native(e))
    "\n    @param title Title of the Merge Request\n    @param source_branch Merge Request's source branch\n    @param target_branch Merge Request's target branch\n    @param state_filter Merge Request's state to filter on\n    "

    def get_mr(self, title, source_branch, target_branch, state_filter):
        mrs = []
        try:
            mrs = self.project.mergerequests.list(search=title, source_branch=source_branch, target_branch=target_branch, state=state_filter)
        except gitlab.exceptions.GitlabGetError as e:
            self._module.fail_json(msg='Failed to list the Merge Request: %s' % to_native(e))
        if len(mrs) > 1:
            self._module.fail_json(msg='Multiple Merge Requests matched search criteria.')
        if len(mrs) == 1:
            try:
                return self.project.mergerequests.get(id=mrs[0].iid)
            except gitlab.exceptions.GitlabGetError as e:
                self._module.fail_json(msg='Failed to get the Merge Request: %s' % to_native(e))
    '\n    @param username Name of the user\n    '

    def get_user(self, username):
        users = []
        try:
            users = [user for user in self.project.users.list(username=username, all=True) if user.username == username]
        except gitlab.exceptions.GitlabGetError as e:
            self._module.fail_json(msg='Failed to list the users: %s' % to_native(e))
        if len(users) > 1:
            self._module.fail_json(msg='Multiple Users matched search criteria.')
        elif len(users) < 1:
            self._module.fail_json(msg='No User matched search criteria.')
        else:
            return users[0]
    '\n    @param users List of usernames\n    '

    def get_user_ids(self, users):
        return [self.get_user(user).id for user in users]
    '\n    @param options Options of the Merge Request\n    '

    def create_mr(self, options):
        if self._module.check_mode:
            self._module.exit_json(changed=True, msg='Successfully created the Merge Request %s' % options['title'])
        try:
            return self.project.mergerequests.create(options)
        except gitlab.exceptions.GitlabCreateError as e:
            self._module.fail_json(msg='Failed to create Merge Request: %s ' % to_native(e))
    '\n    @param mr Merge Request object to delete\n    '

    def delete_mr(self, mr):
        if self._module.check_mode:
            self._module.exit_json(changed=True, msg='Successfully deleted the Merge Request %s' % mr['title'])
        try:
            return mr.delete()
        except gitlab.exceptions.GitlabDeleteError as e:
            self._module.fail_json(msg='Failed to delete Merge Request: %s ' % to_native(e))
    '\n    @param mr Merge Request object to update\n    '

    def update_mr(self, mr, options):
        if self._module.check_mode:
            self._module.exit_json(changed=True, msg='Successfully updated the Merge Request %s' % mr['title'])
        try:
            return self.project.mergerequests.update(mr.iid, options)
        except gitlab.exceptions.GitlabUpdateError as e:
            self._module.fail_json(msg='Failed to update Merge Request: %s ' % to_native(e))
    '\n    @param mr Merge Request object to evaluate\n    @param options New options to update MR with\n    '

    def mr_has_changed(self, mr, options):
        for key, value in options.items():
            if value is not None:
                if key == 'remove_source_branch':
                    key = 'force_remove_source_branch'
                if key == 'assignee_ids':
                    if options[key] != sorted([user['id'] for user in getattr(mr, 'assignees')]):
                        return True
                elif key == 'reviewer_ids':
                    if options[key] != sorted([user['id'] for user in getattr(mr, 'reviewers')]):
                        return True
                elif key == 'labels':
                    if options[key] != sorted(getattr(mr, key)):
                        return True
                elif getattr(mr, key) != value:
                    return True
        return False