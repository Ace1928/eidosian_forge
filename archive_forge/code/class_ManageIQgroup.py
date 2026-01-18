from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.manageiq import ManageIQ, manageiq_argument_spec
class ManageIQgroup(object):
    """
        Object to execute group management operations in manageiq.
    """

    def __init__(self, manageiq):
        self.manageiq = manageiq
        self.module = self.manageiq.module
        self.api_url = self.manageiq.api_url
        self.client = self.manageiq.client

    def group(self, description):
        """ Search for group object by description.
        Returns:
            the group, or None if group was not found.
        """
        groups = self.client.collections.groups.find_by(description=description)
        if len(groups) == 0:
            return None
        else:
            return groups[0]

    def tenant(self, tenant_id, tenant_name):
        """ Search for tenant entity by name or id
        Returns:
            the tenant entity, None if no id or name was supplied
        """
        if tenant_id:
            tenant = self.client.get_entity('tenants', tenant_id)
            if not tenant:
                self.module.fail_json(msg="Tenant with id '%s' not found in manageiq" % str(tenant_id))
            return tenant
        elif tenant_name:
            tenant_res = self.client.collections.tenants.find_by(name=tenant_name)
            if not tenant_res:
                self.module.fail_json(msg="Tenant '%s' not found in manageiq" % tenant_name)
            if len(tenant_res) > 1:
                self.module.fail_json(msg="Multiple tenants found in manageiq with name '%s" % tenant_name)
            tenant = tenant_res[0]
            return tenant
        else:
            return None

    def role(self, role_id, role_name):
        """ Search for a role object by name or id.
        Returns:
            the role entity, None no id or name was supplied

            the role, or send a module Fail signal if role not found.
        """
        if role_id:
            role = self.client.get_entity('roles', role_id)
            if not role:
                self.module.fail_json(msg="Role with id '%s' not found in manageiq" % str(role_id))
            return role
        elif role_name:
            role_res = self.client.collections.roles.find_by(name=role_name)
            if not role_res:
                self.module.fail_json(msg="Role '%s' not found in manageiq" % role_name)
            if len(role_res) > 1:
                self.module.fail_json(msg="Multiple roles found in manageiq with name '%s" % role_name)
            return role_res[0]
        else:
            return None

    @staticmethod
    def merge_dict_values(norm_current_values, norm_updated_values):
        """ Create an merged update object for manageiq group filters.

            The input dict contain the tag values per category.
            If the new values contain the category, all tags for that category are replaced
            If the new values do not contain the category, the existing tags are kept

        Returns:
            the nested array with the merged values, used in the update post body
        """
        if norm_current_values and (not norm_updated_values):
            return norm_current_values
        if not norm_current_values and norm_updated_values:
            return norm_updated_values
        res = norm_current_values.copy()
        res.update(norm_updated_values)
        return res

    def delete_group(self, group):
        """ Deletes a group from manageiq.

        Returns:
            a dict of:
            changed: boolean indicating if the entity was updated.
            msg: a short message describing the operation executed.
        """
        try:
            url = '%s/groups/%s' % (self.api_url, group['id'])
            result = self.client.post(url, action='delete')
        except Exception as e:
            self.module.fail_json(msg='failed to delete group %s: %s' % (group['description'], str(e)))
        if result['success'] is False:
            self.module.fail_json(msg=result['message'])
        return dict(changed=True, msg='deleted group %s with id %s' % (group['description'], group['id']))

    def edit_group(self, group, description, role, tenant, norm_managed_filters, managed_filters_merge_mode, belongsto_filters, belongsto_filters_merge_mode):
        """ Edit a manageiq group.

        Returns:
            a dict of:
            changed: boolean indicating if the entity was updated.
            msg: a short message describing the operation executed.
        """
        if role or norm_managed_filters or belongsto_filters:
            group.reload(attributes=['miq_user_role_name', 'entitlement'])
        try:
            current_role = group['miq_user_role_name']
        except AttributeError:
            current_role = None
        changed = False
        resource = {}
        if description and group['description'] != description:
            resource['description'] = description
            changed = True
        if tenant and group['tenant_id'] != tenant['id']:
            resource['tenant'] = dict(id=tenant['id'])
            changed = True
        if role and current_role != role['name']:
            resource['role'] = dict(id=role['id'])
            changed = True
        if norm_managed_filters or belongsto_filters:
            entitlement = group['entitlement']
            if 'filters' not in entitlement:
                managed_tag_filters_post_body = self.normalized_managed_tag_filters_to_miq(norm_managed_filters)
                resource['filters'] = {'managed': managed_tag_filters_post_body, 'belongsto': belongsto_filters}
                changed = True
            else:
                current_filters = entitlement['filters']
                new_filters = self.edit_group_edit_filters(current_filters, norm_managed_filters, managed_filters_merge_mode, belongsto_filters, belongsto_filters_merge_mode)
                if new_filters:
                    resource['filters'] = new_filters
                    changed = True
        if not changed:
            return dict(changed=False, msg='group %s is not changed.' % group['description'])
        try:
            self.client.post(group['href'], action='edit', resource=resource)
            changed = True
        except Exception as e:
            self.module.fail_json(msg='failed to update group %s: %s' % (group['name'], str(e)))
        return dict(changed=changed, msg='successfully updated the group %s with id %s' % (group['description'], group['id']))

    def edit_group_edit_filters(self, current_filters, norm_managed_filters, managed_filters_merge_mode, belongsto_filters, belongsto_filters_merge_mode):
        """ Edit a manageiq group filters.

        Returns:
            None if no the group was not updated
            If the group was updated the post body part for updating the group
        """
        filters_updated = False
        new_filters_resource = {}
        current_belongsto_set = current_filters.get('belongsto', set())
        if belongsto_filters:
            new_belongsto_set = set(belongsto_filters)
        else:
            new_belongsto_set = set()
        if current_belongsto_set == new_belongsto_set:
            new_filters_resource['belongsto'] = current_filters['belongsto']
        else:
            if belongsto_filters_merge_mode == 'merge':
                current_belongsto_set.update(new_belongsto_set)
                new_filters_resource['belongsto'] = list(current_belongsto_set)
            else:
                new_filters_resource['belongsto'] = list(new_belongsto_set)
            filters_updated = True
        norm_current_filters = self.manageiq_filters_to_sorted_dict(current_filters)
        if norm_current_filters == norm_managed_filters:
            if 'managed' in current_filters:
                new_filters_resource['managed'] = current_filters['managed']
        else:
            if managed_filters_merge_mode == 'merge':
                merged_dict = self.merge_dict_values(norm_current_filters, norm_managed_filters)
                new_filters_resource['managed'] = self.normalized_managed_tag_filters_to_miq(merged_dict)
            else:
                new_filters_resource['managed'] = self.normalized_managed_tag_filters_to_miq(norm_managed_filters)
            filters_updated = True
        if not filters_updated:
            return None
        return new_filters_resource

    def create_group(self, description, role, tenant, norm_managed_filters, belongsto_filters):
        """ Creates the group in manageiq.

        Returns:
            the created group id, name, created_on timestamp,
            updated_on timestamp.
        """
        for key, value in dict(description=description).items():
            if value in (None, ''):
                self.module.fail_json(msg='missing required argument: %s' % key)
        url = '%s/groups' % self.api_url
        resource = {'description': description}
        if role is not None:
            resource['role'] = dict(id=role['id'])
        if tenant is not None:
            resource['tenant'] = dict(id=tenant['id'])
        if norm_managed_filters or belongsto_filters:
            managed_tag_filters_post_body = self.normalized_managed_tag_filters_to_miq(norm_managed_filters)
            resource['filters'] = {'managed': managed_tag_filters_post_body, 'belongsto': belongsto_filters}
        try:
            result = self.client.post(url, action='create', resource=resource)
        except Exception as e:
            self.module.fail_json(msg='failed to create group %s: %s' % (description, str(e)))
        return dict(changed=True, msg='successfully created group %s' % description, group_id=result['results'][0]['id'])

    @staticmethod
    def normalized_managed_tag_filters_to_miq(norm_managed_filters):
        if not norm_managed_filters:
            return None
        return list(norm_managed_filters.values())

    @staticmethod
    def manageiq_filters_to_sorted_dict(current_filters):
        current_managed_filters = current_filters.get('managed')
        if not current_managed_filters:
            return None
        res = {}
        for tag_list in current_managed_filters:
            tag_list.sort()
            key = tag_list[0].split('/')[2]
            res[key] = tag_list
        return res

    @staticmethod
    def normalize_user_managed_filters_to_sorted_dict(managed_filters, module):
        if not managed_filters:
            return None
        res = {}
        for cat_key in managed_filters:
            cat_array = []
            if not isinstance(managed_filters[cat_key], list):
                module.fail_json(msg='Entry "{0}" of managed_filters must be a list!'.format(cat_key))
            for tags in managed_filters[cat_key]:
                miq_managed_tag = '/managed/' + cat_key + '/' + tags
                cat_array.append(miq_managed_tag)
            if cat_array:
                cat_array.sort()
                res[cat_key] = cat_array
        return res

    @staticmethod
    def create_result_group(group):
        """ Creates the ansible result object from a manageiq group entity

        Returns:
            a dict with the group id, description, role, tenant, filters, group_type, created_on, updated_on
        """
        try:
            role_name = group['miq_user_role_name']
        except AttributeError:
            role_name = None
        managed_filters = None
        belongsto_filters = None
        if 'filters' in group['entitlement']:
            filters = group['entitlement']['filters']
            belongsto_filters = filters.get('belongsto')
            group_managed_filters = filters.get('managed')
            if group_managed_filters:
                managed_filters = {}
                for tag_list in group_managed_filters:
                    key = tag_list[0].split('/')[2]
                    tags = []
                    for t in tag_list:
                        tags.append(t.split('/')[3])
                    managed_filters[key] = tags
        return dict(id=group['id'], description=group['description'], role=role_name, tenant=group['tenant']['name'], managed_filters=managed_filters, belongsto_filters=belongsto_filters, group_type=group['group_type'], created_on=group['created_on'], updated_on=group['updated_on'])