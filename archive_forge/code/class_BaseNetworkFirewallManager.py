import time
from copy import deepcopy
from ansible.module_utils._text import to_text
from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.six import string_types
from ansible_collections.amazon.aws.plugins.module_utils.arn import parse_aws_arn
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.community.aws.plugins.module_utils.base import BaseResourceManager
from ansible_collections.community.aws.plugins.module_utils.base import BaseWaiterFactory
from ansible_collections.community.aws.plugins.module_utils.base import Boto3Mixin
from ansible_collections.community.aws.plugins.module_utils.ec2 import BaseEc2Manager
class BaseNetworkFirewallManager(BaseResourceManager):

    def __init__(self, module):
        """
        Parameters:
            module (AnsibleAWSModule): An Ansible module.
        """
        super().__init__(module)
        self.client = self._create_client()
        self._preupdate_metadata = dict()
        self._metadata_updates = dict()
        self._tagging_updates = dict()

    @Boto3Mixin.aws_error_handler('connect to AWS')
    def _create_client(self, client_name='network-firewall'):
        client = self.module.client(client_name, retry_decorator=AWSRetry.jittered_backoff())
        return client

    def _get_id_params(self):
        return dict()

    def _check_updates_pending(self):
        if self._metadata_updates:
            return True
        return super(BaseNetworkFirewallManager, self)._check_updates_pending()

    def _merge_metadata_changes(self, filter_immutable=True):
        """
        Merges the contents of the 'pre_update' metadata variables
        with the pending updates
        """
        metadata = deepcopy(self._preupdate_metadata)
        metadata.update(self._metadata_updates)
        if filter_immutable:
            metadata = self._filter_immutable_metadata_attributes(metadata)
        return metadata

    def _merge_changes(self, filter_metadata=True):
        """
        Merges the contents of the 'pre_update' resource and metadata variables
        with the pending updates
        """
        metadata = self._merge_metadata_changes(filter_metadata)
        resource = self._merge_resource_changes()
        return (metadata, resource)

    def _filter_immutable_metadata_attributes(self, metadata):
        """
        Removes information from the metadata which can't be updated.
        Returns a *copy* of the metadata dictionary.
        """
        meta = deepcopy(metadata)
        meta.pop('LastModifiedTime', None)
        return meta

    def _flush_create(self):
        changed = super(BaseNetworkFirewallManager, self)._flush_create()
        self._metadata_updates = dict()
        return changed

    def _flush_update(self):
        changed = super(BaseNetworkFirewallManager, self)._flush_update()
        self._metadata_updates = dict()
        return changed

    @BaseResourceManager.aws_error_handler('set tags on resource')
    def _add_tags(self, **params):
        self.client.tag_resource(aws_retry=True, **params)
        return True

    @BaseResourceManager.aws_error_handler('unset tags on resource')
    def _remove_tags(self, **params):
        self.client.untag_resource(aws_retry=True, **params)
        return True

    def _get_preupdate_arn(self):
        return self._preupdate_metadata.get('Arn')

    def _set_metadata_value(self, key, value, description=None, immutable=False):
        if value is None:
            return False
        if value == self._get_metadata_value(key):
            return False
        if immutable and self.original_resource:
            if description is None:
                description = key
            self.module.fail_json(msg=f'{description} can not be updated after creation')
        self._metadata_updates[key] = value
        self.changed = True
        return True

    def _get_metadata_value(self, key, default=None):
        return self._metadata_updates.get(key, self._preupdate_metadata.get(key, default))

    def _set_tag_values(self, desired_tags):
        return self._set_metadata_value('Tags', ansible_dict_to_boto3_tag_list(desired_tags))

    def _get_tag_values(self):
        return self._get_metadata_value('Tags', [])

    def _flush_tagging(self):
        changed = False
        tags_to_add = self._tagging_updates.get('add')
        tags_to_remove = self._tagging_updates.get('remove')
        resource_arn = self._get_preupdate_arn()
        if not resource_arn:
            return False
        if tags_to_add:
            changed = True
            tags = ansible_dict_to_boto3_tag_list(tags_to_add)
            if not self.module.check_mode:
                self._add_tags(ResourceArn=resource_arn, Tags=tags)
        if tags_to_remove:
            changed = True
            if not self.module.check_mode:
                self._remove_tags(ResourceArn=resource_arn, TagKeys=tags_to_remove)
        return changed

    def set_tags(self, tags, purge_tags):
        if tags is None:
            return False
        changed = False
        current_tags = boto3_tag_list_to_ansible_dict(self._get_tag_values())
        if purge_tags:
            desired_tags = deepcopy(tags)
        else:
            desired_tags = deepcopy(current_tags)
            desired_tags.update(tags)
        tags_to_add, tags_to_remove = compare_aws_tags(current_tags, tags, purge_tags)
        if tags_to_add:
            self._tagging_updates['add'] = tags_to_add
            changed = True
        if tags_to_remove:
            self._tagging_updates['remove'] = tags_to_remove
            changed = True
        if changed:
            return self._set_tag_values(desired_tags)
        return False