from ansible.errors import AnsibleError
from ansible.module_utils._text import to_native
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.plugin_utils.inventory import AWSInventoryBase
def _add_details_to_hosts(connection, hosts, strict):
    for host in hosts:
        detail = None
        resource_id = host['BrokerId']
        try:
            detail = connection.describe_broker(BrokerId=resource_id)
        except is_boto3_error_code('AccessDenied') as e:
            if not strict:
                pass
            else:
                raise AnsibleError(f'Failed to query MQ: {to_native(e)}')
        except (botocore.exceptions.BotoCoreError, botocore.exceptions.ClientError) as e:
            raise AnsibleError(f'Failed to query MQ: {to_native(e)}')
        if detail:
            host['Tags'] = _get_broker_host_tags(detail)
            for attr in broker_attr:
                if attr in detail:
                    host[attr] = detail[attr]