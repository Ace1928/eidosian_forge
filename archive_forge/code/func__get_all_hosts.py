from ansible.errors import AnsibleError
from ansible.module_utils._text import to_native
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.plugin_utils.inventory import AWSInventoryBase
def _get_all_hosts(self, regions, strict, statuses):
    """
        :param regions: a list of regions in which to describe hosts
        :param strict: a boolean determining whether to fail or ignore 403 error codes
        :param statuses: a list of statuses that the returned hosts should match
        :return A list of host dictionaries
        """
    all_instances = []
    for connection, _region in self.all_clients('mq'):
        paginator = connection.get_paginator('list_brokers')
        all_instances.extend(self._get_broker_hosts(connection, strict)(paginator.paginate().build_full_result))
    sorted_hosts = list(sorted(all_instances, key=lambda x: x['BrokerName']))
    return _find_hosts_matching_statuses(sorted_hosts, statuses)