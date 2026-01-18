from __future__ import absolute_import, division, print_function
import copy
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common import AZURE_SUCCESS_STATE, AzureRMModuleBase
from ansible.module_utils._text import to_native
def account_obj_to_dict(self, account_obj, blob_mgmt_props=None, blob_client_props=None):
    account_dict = dict(id=account_obj.id, name=account_obj.name, location=account_obj.location, failover_in_progress=account_obj.failover_in_progress if account_obj.failover_in_progress is not None else False, resource_group=self.resource_group, type=account_obj.type, access_tier=account_obj.access_tier, sku_tier=account_obj.sku.tier, sku_name=account_obj.sku.name, provisioning_state=account_obj.provisioning_state, secondary_location=account_obj.secondary_location, status_of_primary=account_obj.status_of_primary, status_of_secondary=account_obj.status_of_secondary, primary_location=account_obj.primary_location, https_only=account_obj.enable_https_traffic_only, minimum_tls_version=account_obj.minimum_tls_version, public_network_access=account_obj.public_network_access, allow_blob_public_access=account_obj.allow_blob_public_access, network_acls=account_obj.network_rule_set, is_hns_enabled=account_obj.is_hns_enabled if account_obj.is_hns_enabled else False, static_website=dict(enabled=False, index_document=None, error_document404_path=None))
    account_dict['custom_domain'] = None
    if account_obj.custom_domain:
        account_dict['custom_domain'] = dict(name=account_obj.custom_domain.name, use_sub_domain=account_obj.custom_domain.use_sub_domain)
    account_dict['primary_endpoints'] = None
    if account_obj.primary_endpoints:
        account_dict['primary_endpoints'] = dict(blob=account_obj.primary_endpoints.blob, queue=account_obj.primary_endpoints.queue, table=account_obj.primary_endpoints.table)
    account_dict['secondary_endpoints'] = None
    if account_obj.secondary_endpoints:
        account_dict['secondary_endpoints'] = dict(blob=account_obj.secondary_endpoints.blob, queue=account_obj.secondary_endpoints.queue, table=account_obj.secondary_endpoints.table)
    account_dict['tags'] = None
    if account_obj.tags:
        account_dict['tags'] = account_obj.tags
    if blob_mgmt_props and blob_mgmt_props.cors and blob_mgmt_props.cors.cors_rules:
        account_dict['blob_cors'] = [dict(allowed_origins=[to_native(y) for y in x.allowed_origins], allowed_methods=[to_native(y) for y in x.allowed_methods], max_age_in_seconds=x.max_age_in_seconds, exposed_headers=[to_native(y) for y in x.exposed_headers], allowed_headers=[to_native(y) for y in x.allowed_headers]) for x in blob_mgmt_props.cors.cors_rules]
    if blob_client_props and blob_client_props['static_website']:
        static_website = blob_client_props['static_website']
        account_dict['static_website'] = dict(enabled=static_website.enabled, index_document=static_website.index_document, error_document404_path=static_website.error_document404_path)
    account_dict['network_acls'] = None
    if account_obj.network_rule_set:
        account_dict['network_acls'] = dict(bypass=account_obj.network_rule_set.bypass, default_action=account_obj.network_rule_set.default_action)
        account_dict['network_acls']['virtual_network_rules'] = []
        if account_obj.network_rule_set.virtual_network_rules:
            for rule in account_obj.network_rule_set.virtual_network_rules:
                account_dict['network_acls']['virtual_network_rules'].append(dict(id=rule.virtual_network_resource_id, action=rule.action))
        account_dict['network_acls']['ip_rules'] = []
        if account_obj.network_rule_set.ip_rules:
            for rule in account_obj.network_rule_set.ip_rules:
                account_dict['network_acls']['ip_rules'].append(dict(value=rule.ip_address_or_range, action=rule.action))
        account_dict['encryption'] = dict()
        if account_obj.encryption:
            account_dict['encryption']['require_infrastructure_encryption'] = account_obj.encryption.require_infrastructure_encryption
            account_dict['encryption']['key_source'] = account_obj.encryption.key_source
            if account_obj.encryption.services:
                account_dict['encryption']['services'] = dict()
                if account_obj.encryption.services.file:
                    account_dict['encryption']['services']['file'] = dict(enabled=True)
                if account_obj.encryption.services.table:
                    account_dict['encryption']['services']['table'] = dict(enabled=True)
                if account_obj.encryption.services.queue:
                    account_dict['encryption']['services']['queue'] = dict(enabled=True)
                if account_obj.encryption.services.blob:
                    account_dict['encryption']['services']['blob'] = dict(enabled=True)
    return account_dict