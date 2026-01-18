from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.manageiq import ManageIQ, manageiq_argument_spec
class ManageIQTenant(object):
    """
        Object to execute tenant management operations in manageiq.
    """

    def __init__(self, manageiq):
        self.manageiq = manageiq
        self.module = self.manageiq.module
        self.api_url = self.manageiq.api_url
        self.client = self.manageiq.client

    def tenant(self, name, parent_id, parent):
        """ Search for tenant object by name and parent_id or parent
            or the root tenant if no parent or parent_id is supplied.
        Returns:
            the parent tenant, None for the root tenant
            the tenant or None if tenant was not found.
        """
        if parent_id:
            parent_tenant_res = self.client.collections.tenants.find_by(id=parent_id)
            if not parent_tenant_res:
                self.module.fail_json(msg="Parent tenant with id '%s' not found in manageiq" % str(parent_id))
            parent_tenant = parent_tenant_res[0]
            tenants = self.client.collections.tenants.find_by(name=name)
            for tenant in tenants:
                try:
                    ancestry = tenant['ancestry']
                except AttributeError:
                    ancestry = None
                if ancestry:
                    tenant_parent_id = int(ancestry.split('/')[-1])
                    if int(tenant_parent_id) == parent_id:
                        return (parent_tenant, tenant)
            return (parent_tenant, None)
        elif parent:
            parent_tenant_res = self.client.collections.tenants.find_by(name=parent)
            if not parent_tenant_res:
                self.module.fail_json(msg="Parent tenant '%s' not found in manageiq" % parent)
            if len(parent_tenant_res) > 1:
                self.module.fail_json(msg="Multiple parent tenants not found in manageiq with name '%s" % parent)
            parent_tenant = parent_tenant_res[0]
            parent_id = int(parent_tenant['id'])
            tenants = self.client.collections.tenants.find_by(name=name)
            for tenant in tenants:
                try:
                    ancestry = tenant['ancestry']
                except AttributeError:
                    ancestry = None
                if ancestry:
                    tenant_parent_id = int(ancestry.split('/')[-1])
                    if tenant_parent_id == parent_id:
                        return (parent_tenant, tenant)
            return (parent_tenant, None)
        else:
            return (None, self.client.collections.tenants.find_by(ancestry=None)[0])

    def compare_tenant(self, tenant, name, description):
        """ Compare tenant fields with new field values.

        Returns:
            false if tenant fields have some difference from new fields, true o/w.
        """
        found_difference = name and tenant['name'] != name or (description and tenant['description'] != description)
        return not found_difference

    def delete_tenant(self, tenant):
        """ Deletes a tenant from manageiq.

        Returns:
            dict with `msg` and `changed`
        """
        try:
            url = '%s/tenants/%s' % (self.api_url, tenant['id'])
            result = self.client.post(url, action='delete')
        except Exception as e:
            self.module.fail_json(msg='failed to delete tenant %s: %s' % (tenant['name'], str(e)))
        if result['success'] is False:
            self.module.fail_json(msg=result['message'])
        return dict(changed=True, msg=result['message'])

    def edit_tenant(self, tenant, name, description):
        """ Edit a manageiq tenant.

        Returns:
            dict with `msg` and `changed`
        """
        resource = dict(name=name, description=description, use_config_for_attributes=False)
        if self.compare_tenant(tenant, name, description):
            return dict(changed=False, msg='tenant %s is not changed.' % tenant['name'], tenant=tenant['_data'])
        try:
            result = self.client.post(tenant['href'], action='edit', resource=resource)
        except Exception as e:
            self.module.fail_json(msg='failed to update tenant %s: %s' % (tenant['name'], str(e)))
        return dict(changed=True, msg='successfully updated the tenant with id %s' % tenant['id'])

    def create_tenant(self, name, description, parent_tenant):
        """ Creates the tenant in manageiq.

        Returns:
            dict with `msg`, `changed` and `tenant_id`
        """
        parent_id = parent_tenant['id']
        for key, value in dict(name=name, description=description, parent_id=parent_id).items():
            if value in (None, ''):
                self.module.fail_json(msg='missing required argument: %s' % key)
        url = '%s/tenants' % self.api_url
        resource = {'name': name, 'description': description, 'parent': {'id': parent_id}}
        try:
            result = self.client.post(url, action='create', resource=resource)
            tenant_id = result['results'][0]['id']
        except Exception as e:
            self.module.fail_json(msg='failed to create tenant %s: %s' % (name, str(e)))
        return dict(changed=True, msg="successfully created tenant '%s' with id '%s'" % (name, tenant_id), tenant_id=tenant_id)

    def tenant_quota(self, tenant, quota_key):
        """ Search for tenant quota object by tenant and quota_key.
        Returns:
            the quota for the tenant, or None if the tenant quota was not found.
        """
        tenant_quotas = self.client.get('%s/quotas?expand=resources&filter[]=name=%s' % (tenant['href'], quota_key))
        return tenant_quotas['resources']

    def tenant_quotas(self, tenant):
        """ Search for tenant quotas object by tenant.
        Returns:
            the quotas for the tenant, or None if no tenant quotas were not found.
        """
        tenant_quotas = self.client.get('%s/quotas?expand=resources' % tenant['href'])
        return tenant_quotas['resources']

    def update_tenant_quotas(self, tenant, quotas):
        """ Creates the tenant quotas in manageiq.

        Returns:
            dict with `msg` and `changed`
        """
        changed = False
        messages = []
        for quota_key, quota_value in quotas.items():
            current_quota_filtered = self.tenant_quota(tenant, quota_key)
            if current_quota_filtered:
                current_quota = current_quota_filtered[0]
            else:
                current_quota = None
            if quota_value:
                if quota_key in ['storage_allocated', 'mem_allocated']:
                    quota_value_int = int(quota_value) * 1024 * 1024 * 1024
                else:
                    quota_value_int = int(quota_value)
                if current_quota:
                    res = self.edit_tenant_quota(tenant, current_quota, quota_key, quota_value_int)
                else:
                    res = self.create_tenant_quota(tenant, quota_key, quota_value_int)
            elif current_quota:
                res = self.delete_tenant_quota(tenant, current_quota)
            else:
                res = dict(changed=False, msg="tenant quota '%s' does not exist" % quota_key)
            if res['changed']:
                changed = True
            messages.append(res['msg'])
        return dict(changed=changed, msg=', '.join(messages))

    def edit_tenant_quota(self, tenant, current_quota, quota_key, quota_value):
        """ Update the tenant quotas in manageiq.

        Returns:
            result
        """
        if current_quota['value'] == quota_value:
            return dict(changed=False, msg='tenant quota %s already has value %s' % (quota_key, quota_value))
        else:
            url = '%s/quotas/%s' % (tenant['href'], current_quota['id'])
            resource = {'value': quota_value}
            try:
                self.client.post(url, action='edit', resource=resource)
            except Exception as e:
                self.module.fail_json(msg='failed to update tenant quota %s: %s' % (quota_key, str(e)))
            return dict(changed=True, msg='successfully updated tenant quota %s' % quota_key)

    def create_tenant_quota(self, tenant, quota_key, quota_value):
        """ Creates the tenant quotas in manageiq.

        Returns:
            result
        """
        url = '%s/quotas' % tenant['href']
        resource = {'name': quota_key, 'value': quota_value}
        try:
            self.client.post(url, action='create', resource=resource)
        except Exception as e:
            self.module.fail_json(msg='failed to create tenant quota %s: %s' % (quota_key, str(e)))
        return dict(changed=True, msg='successfully created tenant quota %s' % quota_key)

    def delete_tenant_quota(self, tenant, quota):
        """ deletes the tenant quotas in manageiq.

        Returns:
            result
        """
        try:
            result = self.client.post(quota['href'], action='delete')
        except Exception as e:
            self.module.fail_json(msg="failed to delete tenant quota '%s': %s" % (quota['name'], str(e)))
        return dict(changed=True, msg=result['message'])

    def create_tenant_response(self, tenant, parent_tenant):
        """ Creates the ansible result object from a manageiq tenant entity

        Returns:
            a dict with the tenant id, name, description, parent id,
            quota's
        """
        tenant_quotas = self.create_tenant_quotas_response(tenant['tenant_quotas'])
        try:
            ancestry = tenant['ancestry']
            tenant_parent_id = ancestry.split('/')[-1]
        except AttributeError:
            tenant_parent_id = None
        return dict(id=tenant['id'], name=tenant['name'], description=tenant['description'], parent_id=tenant_parent_id, quotas=tenant_quotas)

    @staticmethod
    def create_tenant_quotas_response(tenant_quotas):
        """ Creates the ansible result object from a manageiq tenant_quotas entity

        Returns:
            a dict with the applied quotas, name and value
        """
        if not tenant_quotas:
            return {}
        result = {}
        for quota in tenant_quotas:
            if quota['unit'] == 'bytes':
                value = float(quota['value']) / (1024 * 1024 * 1024)
            else:
                value = quota['value']
            result[quota['name']] = value
        return result