from ansible.errors import AnsibleError
from ansible.module_utils._text import to_native
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
from ansible_collections.amazon.aws.plugins.plugin_utils.inventory import AWSInventoryBase
def _add_hosts(self, hosts, group):
    """
        :param hosts: a list of hosts to be added to a group
        :param group: the name of the group to which the hosts belong
        """
    for host in hosts:
        hostname = _get_rds_hostname(host)
        host = camel_dict_to_snake_dict(host, ignore_list=['Tags'])
        host['tags'] = boto3_tag_list_to_ansible_dict(host.get('tags', []))
        if 'availability_zone' in host:
            host['region'] = host['availability_zone'][:-1]
        elif 'availability_zones' in host:
            host['region'] = host['availability_zones'][0][:-1]
        self.inventory.add_host(hostname, group=group)
        hostvars_prefix = self.get_option('hostvars_prefix')
        hostvars_suffix = self.get_option('hostvars_suffix')
        new_vars = dict()
        for hostvar, hostval in host.items():
            if hostvars_prefix:
                hostvar = hostvars_prefix + hostvar
            if hostvars_suffix:
                hostvar = hostvar + hostvars_suffix
            new_vars[hostvar] = hostval
            self.inventory.set_variable(hostname, hostvar, hostval)
        host.update(new_vars)
        strict = self.get_option('strict')
        self._set_composite_vars(self.get_option('compose'), host, hostname, strict=strict)
        self._add_host_to_composed_groups(self.get_option('groups'), host, hostname, strict=strict)
        self._add_host_to_keyed_groups(self.get_option('keyed_groups'), host, hostname, strict=strict)