from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.manageiq import ManageIQ, manageiq_argument_spec
class ManageIQUser(object):
    """
        Object to execute user management operations in manageiq.
    """

    def __init__(self, manageiq):
        self.manageiq = manageiq
        self.module = self.manageiq.module
        self.api_url = self.manageiq.api_url
        self.client = self.manageiq.client

    def group_id(self, description):
        """ Search for group id by group description.

        Returns:
            the group id, or send a module Fail signal if group not found.
        """
        group = self.manageiq.find_collection_resource_by('groups', description=description)
        if not group:
            self.module.fail_json(msg='group %s does not exist in manageiq' % description)
        return group['id']

    def user(self, userid):
        """ Search for user object by userid.

        Returns:
            the user, or None if user not found.
        """
        return self.manageiq.find_collection_resource_by('users', userid=userid)

    def compare_user(self, user, name, group_id, password, email):
        """ Compare user fields with new field values.

        Returns:
            false if user fields have some difference from new fields, true o/w.
        """
        found_difference = name and user['name'] != name or password is not None or (email and user['email'] != email) or (group_id and user['current_group_id'] != group_id)
        return not found_difference

    def delete_user(self, user):
        """ Deletes a user from manageiq.

        Returns:
            a short message describing the operation executed.
        """
        try:
            url = '%s/users/%s' % (self.api_url, user['id'])
            result = self.client.post(url, action='delete')
        except Exception as e:
            self.module.fail_json(msg='failed to delete user %s: %s' % (user['userid'], str(e)))
        return dict(changed=True, msg=result['message'])

    def edit_user(self, user, name, group, password, email):
        """ Edit a user from manageiq.

        Returns:
            a short message describing the operation executed.
        """
        group_id = None
        url = '%s/users/%s' % (self.api_url, user['id'])
        resource = dict(userid=user['userid'])
        if group is not None:
            group_id = self.group_id(group)
            resource['group'] = dict(id=group_id)
        if name is not None:
            resource['name'] = name
        if email is not None:
            resource['email'] = email
        if self.module.params['update_password'] == 'on_create':
            password = None
        if password is not None:
            resource['password'] = password
        if self.compare_user(user, name, group_id, password, email):
            return dict(changed=False, msg='user %s is not changed.' % user['userid'])
        try:
            result = self.client.post(url, action='edit', resource=resource)
        except Exception as e:
            self.module.fail_json(msg='failed to update user %s: %s' % (user['userid'], str(e)))
        return dict(changed=True, msg='successfully updated the user %s: %s' % (user['userid'], result))

    def create_user(self, userid, name, group, password, email):
        """ Creates the user in manageiq.

        Returns:
            the created user id, name, created_on timestamp,
            updated_on timestamp, userid and current_group_id.
        """
        for key, value in dict(name=name, group=group, password=password).items():
            if value in (None, ''):
                self.module.fail_json(msg='missing required argument: %s' % key)
        group_id = self.group_id(group)
        url = '%s/users' % self.api_url
        resource = {'userid': userid, 'name': name, 'password': password, 'group': {'id': group_id}}
        if email is not None:
            resource['email'] = email
        try:
            result = self.client.post(url, action='create', resource=resource)
        except Exception as e:
            self.module.fail_json(msg='failed to create user %s: %s' % (userid, str(e)))
        return dict(changed=True, msg='successfully created the user %s: %s' % (userid, result['results']))