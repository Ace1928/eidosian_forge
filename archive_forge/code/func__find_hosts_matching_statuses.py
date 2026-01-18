from ansible.errors import AnsibleError
from ansible.module_utils._text import to_native
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.plugin_utils.inventory import AWSInventoryBase
def _find_hosts_matching_statuses(hosts, statuses):
    if not statuses:
        statuses = ['RUNNING', 'CREATION_IN_PROGRESS']
    if 'all' in statuses:
        return hosts
    valid_hosts = []
    for host in hosts:
        if host.get('BrokerState') in statuses:
            valid_hosts.append(host)
    return valid_hosts