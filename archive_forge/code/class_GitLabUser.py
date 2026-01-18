from __future__ import absolute_import, division, print_function
from ansible.module_utils.api import basic_auth_argument_spec
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.text.converters import to_native
from ansible_collections.community.general.plugins.module_utils.gitlab import (
class GitLabUser(object):

    def __init__(self, module, gitlab_instance):
        self._module = module
        self._gitlab = gitlab_instance
        self.user_object = None
        self.ACCESS_LEVEL = {'guest': gitlab.const.GUEST_ACCESS, 'reporter': gitlab.const.REPORTER_ACCESS, 'developer': gitlab.const.DEVELOPER_ACCESS, 'master': gitlab.const.MAINTAINER_ACCESS, 'maintainer': gitlab.const.MAINTAINER_ACCESS, 'owner': gitlab.const.OWNER_ACCESS}
    '\n    @param username Username of the user\n    @param options User options\n    '

    def create_or_update_user(self, username, options):
        changed = False
        potentionally_changed = False
        if self.user_object is None:
            user = self.create_user({'name': options['name'], 'username': username, 'password': options['password'], 'reset_password': options['reset_password'], 'email': options['email'], 'skip_confirmation': not options['confirm'], 'admin': options['isadmin'], 'external': options['external'], 'identities': options['identities']})
            changed = True
        else:
            changed, user = self.update_user(self.user_object, {'name': {'value': options['name']}, 'email': {'value': options['email']}, 'is_admin': {'value': options['isadmin'], 'setter': 'admin'}, 'external': {'value': options['external']}, 'identities': {'value': options['identities']}}, {'skip_reconfirmation': {'value': not options['confirm']}, 'password': {'value': options['password']}, 'reset_password': {'value': options['reset_password']}, 'overwrite_identities': {'value': options['overwrite_identities']}})
            potentionally_changed = True
        if options['sshkey_name'] and options['sshkey_file']:
            key_changed = self.add_ssh_key_to_user(user, {'name': options['sshkey_name'], 'file': options['sshkey_file'], 'expires_at': options['sshkey_expires_at']})
            changed = changed or key_changed
        if options['group_path']:
            group_changed = self.assign_user_to_group(user, options['group_path'], options['access_level'])
            changed = changed or group_changed
        self.user_object = user
        if (changed or potentionally_changed) and (not self._module.check_mode):
            try:
                user.save()
            except Exception as e:
                self._module.fail_json(msg='Failed to update user: %s ' % to_native(e))
        if changed:
            if self._module.check_mode:
                self._module.exit_json(changed=True, msg='Successfully created or updated the user %s' % username)
            return True
        else:
            return False
    '\n    @param group User object\n    '

    def get_user_id(self, user):
        if user is not None:
            return user.id
        return None
    '\n    @param user User object\n    @param sshkey_name Name of the ssh key\n    '

    def ssh_key_exists(self, user, sshkey_name):
        return any((k.title == sshkey_name for k in user.keys.list(**list_all_kwargs)))
    '\n    @param user User object\n    @param sshkey Dict containing sshkey infos {"name": "", "file": "", "expires_at": ""}\n    '

    def add_ssh_key_to_user(self, user, sshkey):
        if not self.ssh_key_exists(user, sshkey['name']):
            if self._module.check_mode:
                return True
            try:
                parameter = {'title': sshkey['name'], 'key': sshkey['file']}
                if sshkey['expires_at'] is not None:
                    parameter['expires_at'] = sshkey['expires_at']
                user.keys.create(parameter)
            except gitlab.exceptions.GitlabCreateError as e:
                self._module.fail_json(msg='Failed to assign sshkey to user: %s' % to_native(e))
            return True
        return False
    '\n    @param group Group object\n    @param user_id Id of the user to find\n    '

    def find_member(self, group, user_id):
        try:
            member = group.members.get(user_id)
        except gitlab.exceptions.GitlabGetError:
            return None
        return member
    '\n    @param group Group object\n    @param user_id Id of the user to check\n    '

    def member_exists(self, group, user_id):
        member = self.find_member(group, user_id)
        return member is not None
    '\n    @param group Group object\n    @param user_id Id of the user to check\n    @param access_level GitLab access_level to check\n    '

    def member_as_good_access_level(self, group, user_id, access_level):
        member = self.find_member(group, user_id)
        return member.access_level == access_level
    '\n    @param user User object\n    @param group_path Complete path of the Group including parent group path. <parent_path>/<group_path>\n    @param access_level GitLab access_level to assign\n    '

    def assign_user_to_group(self, user, group_identifier, access_level):
        group = find_group(self._gitlab, group_identifier)
        if self._module.check_mode:
            return True
        if group is None:
            return False
        if self.member_exists(group, self.get_user_id(user)):
            member = self.find_member(group, self.get_user_id(user))
            if not self.member_as_good_access_level(group, member.id, self.ACCESS_LEVEL[access_level]):
                member.access_level = self.ACCESS_LEVEL[access_level]
                member.save()
                return True
        else:
            try:
                group.members.create({'user_id': self.get_user_id(user), 'access_level': self.ACCESS_LEVEL[access_level]})
            except gitlab.exceptions.GitlabCreateError as e:
                self._module.fail_json(msg='Failed to assign user to group: %s' % to_native(e))
            return True
        return False
    '\n    @param user User object\n    @param arguments User attributes\n    '

    def update_user(self, user, arguments, uncheckable_args):
        changed = False
        for arg_key, arg_value in arguments.items():
            av = arg_value['value']
            if av is not None:
                if arg_key == 'identities':
                    changed = self.add_identities(user, av, uncheckable_args['overwrite_identities']['value'])
                elif getattr(user, arg_key) != av:
                    setattr(user, arg_value.get('setter', arg_key), av)
                    changed = True
        for arg_key, arg_value in uncheckable_args.items():
            av = arg_value['value']
            if av is not None:
                setattr(user, arg_value.get('setter', arg_key), av)
        return (changed, user)
    '\n    @param arguments User attributes\n    '

    def create_user(self, arguments):
        if self._module.check_mode:
            return True
        identities = None
        if 'identities' in arguments:
            identities = arguments['identities']
            del arguments['identities']
        try:
            user = self._gitlab.users.create(arguments)
            if identities:
                self.add_identities(user, identities)
        except gitlab.exceptions.GitlabCreateError as e:
            self._module.fail_json(msg='Failed to create user: %s ' % to_native(e))
        return user
    '\n    @param user User object\n    @param identities List of identities to be added/updated\n    @param overwrite_identities Overwrite user identities with identities passed to this module\n    '

    def add_identities(self, user, identities, overwrite_identities=False):
        changed = False
        if overwrite_identities:
            changed = self.delete_identities(user, identities)
        for identity in identities:
            if identity not in user.identities:
                setattr(user, 'provider', identity['provider'])
                setattr(user, 'extern_uid', identity['extern_uid'])
                if not self._module.check_mode:
                    user.save()
                changed = True
        return changed
    '\n    @param user User object\n    @param identities List of identities to be added/updated\n    '

    def delete_identities(self, user, identities):
        changed = False
        for identity in user.identities:
            if identity not in identities:
                if not self._module.check_mode:
                    user.identityproviders.delete(identity['provider'])
                changed = True
        return changed
    '\n    @param username Username of the user\n    '

    def find_user(self, username):
        return next((user for user in self._gitlab.users.list(search=username, **list_all_kwargs) if user.username == username), None)
    '\n    @param username Username of the user\n    '

    def exists_user(self, username):
        user = self.find_user(username)
        if user:
            self.user_object = user
            return True
        return False
    '\n    @param username Username of the user\n    '

    def is_active(self, username):
        user = self.find_user(username)
        return user.attributes['state'] == 'active'

    def delete_user(self):
        if self._module.check_mode:
            return True
        user = self.user_object
        return user.delete()

    def block_user(self):
        if self._module.check_mode:
            return True
        user = self.user_object
        return user.block()

    def unblock_user(self):
        if self._module.check_mode:
            return True
        user = self.user_object
        return user.unblock()