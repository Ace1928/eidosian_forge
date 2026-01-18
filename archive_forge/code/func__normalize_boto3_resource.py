from copy import deepcopy
from functools import wraps
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
def _normalize_boto3_resource(self, resource, add_tags=False):
    """
        Performs common boto3 resource to Ansible resource conversion.
        `resource['Tags']` will by default be converted from the boto3 tag list
        format to a simple dictionary.
        Parameters:
          resource (dict): The boto3 style resource to convert to the normal Ansible
                           format (snake_case).
          add_tags (bool): When `true`, if a resource does not have 'Tags' property
                           the returned resource will have tags set to an empty
                           dictionary.
        """
    if resource is None:
        return None
    tags = resource.get('Tags', None)
    if tags:
        tags = boto3_tag_list_to_ansible_dict(tags)
    elif add_tags or tags is not None:
        tags = {}
    normalized_resource = camel_dict_to_snake_dict(resource)
    if tags is not None:
        normalized_resource['tags'] = tags
    return normalized_resource