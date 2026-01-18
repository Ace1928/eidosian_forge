from ansible.errors import AnsibleError
from ansible.module_utils._text import to_native
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
from ansible_collections.amazon.aws.plugins.plugin_utils.inventory import AWSInventoryBase
def _format_inventory(self, hosts):
    results = {'_meta': {'hostvars': {}}}
    group = 'aws_rds'
    results[group] = {'hosts': []}
    for host in hosts:
        hostname = _get_rds_hostname(host)
        results[group]['hosts'].append(hostname)
        h = self.inventory.get_host(hostname)
        results['_meta']['hostvars'][h.name] = h.vars
    return results