import re
from ansible.module_utils._text import to_text
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
from ansible_collections.amazon.aws.plugins.plugin_utils.inventory import AWSInventoryBase
def _get_all_hostnames(self, instance, hostnames):
    """
        :param instance: an instance dict returned by boto3 ec2 describe_instances()
        :param hostnames: a list of hostname destination variables
        :return all the candidats matching the expectation
        """
    if not hostnames:
        hostnames = ['dns-name', 'private-dns-name']
    hostname = None
    hostname_list = []
    for preference in hostnames:
        if isinstance(preference, dict):
            if 'name' not in preference:
                self.fail_aws("A 'name' key must be defined in a hostnames dictionary.")
            hostname = self._get_all_hostnames(instance, [preference['name']])
            hostname_from_prefix = None
            if 'prefix' in preference:
                hostname_from_prefix = self._get_all_hostnames(instance, [preference['prefix']])
            separator = preference.get('separator', '_')
            if hostname and hostname_from_prefix and ('prefix' in preference):
                hostname = hostname_from_prefix[0] + separator + hostname[0]
        elif preference.startswith('tag:'):
            hostname = _get_tag_hostname(preference, instance)
        else:
            hostname = _get_boto_attr_chain(preference, instance)
        if hostname:
            if isinstance(hostname, list):
                for host in hostname:
                    hostname_list.append(self._sanitize_hostname(host))
            elif isinstance(hostname, str):
                hostname_list.append(self._sanitize_hostname(hostname))
    return hostname_list