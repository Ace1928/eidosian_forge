from __future__ import absolute_import, division, print_function
from datetime import datetime
from ansible.module_utils.api import basic_auth_argument_spec
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.general.plugins.module_utils.gitlab import (
class GitLabProjectAccessToken(object):

    def __init__(self, module, gitlab_instance):
        self._module = module
        self._gitlab = gitlab_instance
        self.access_token_object = None
    '\n    @param project Project Object\n    @param arguments Attributes of the access_token\n    '

    def create_access_token(self, project, arguments):
        changed = False
        if self._module.check_mode:
            return True
        try:
            self.access_token_object = project.access_tokens.create(arguments)
            changed = True
        except gitlab.exceptions.GitlabCreateError as e:
            self._module.fail_json(msg='Failed to create access token: %s ' % to_native(e))
        return changed
    '\n    @param project Project object\n    @param name of the access token\n    '

    def find_access_token(self, project, name):
        access_tokens = project.access_tokens.list(all=True)
        for access_token in access_tokens:
            if access_token.name == name:
                self.access_token_object = access_token
                return False
        return False

    def revoke_access_token(self):
        if self._module.check_mode:
            return True
        changed = False
        try:
            self.access_token_object.delete()
            changed = True
        except gitlab.exceptions.GitlabCreateError as e:
            self._module.fail_json(msg='Failed to revoke access token: %s ' % to_native(e))
        return changed

    def access_tokens_equal(self):
        if self.access_token_object.name != self._module.params['name']:
            return False
        if self.access_token_object.scopes != self._module.params['scopes']:
            return False
        if self.access_token_object.access_level != ACCESS_LEVELS[self._module.params['access_level']]:
            return False
        if self.access_token_object.expires_at != self._module.params['expires_at']:
            return False
        return True