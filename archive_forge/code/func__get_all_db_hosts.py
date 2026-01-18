from ansible.errors import AnsibleError
from ansible.module_utils._text import to_native
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
from ansible_collections.amazon.aws.plugins.plugin_utils.inventory import AWSInventoryBase
def _get_all_db_hosts(self, regions, instance_filters, cluster_filters, strict, statuses, gather_clusters=False):
    """
        :param regions: a list of regions in which to describe hosts
        :param instance_filters: a list of boto3 filter dictionaries
        :param cluster_filters: a list of boto3 filter dictionaries
        :param strict: a boolean determining whether to fail or ignore 403 error codes
        :param statuses: a list of statuses that the returned hosts should match
        :return A list of host dictionaries
        """
    all_instances = []
    all_clusters = []
    for connection, _region in self.all_clients('rds'):
        all_instances += _describe_db_instances(connection, instance_filters, strict=strict)
        if gather_clusters:
            all_clusters += _describe_db_clusters(connection, cluster_filters, strict=strict)
    sorted_hosts = list(sorted(all_instances, key=lambda x: x['DBInstanceIdentifier']) + sorted(all_clusters, key=lambda x: x['DBClusterIdentifier']))
    return _find_hosts_with_valid_statuses(sorted_hosts, statuses)