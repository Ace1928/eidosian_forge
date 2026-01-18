from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
from ansible_collections.community.aws.plugins.module_utils.wafv2 import describe_wafv2_tags
from ansible_collections.community.aws.plugins.module_utils.wafv2 import ensure_wafv2_tags
class IpSet:

    def __init__(self, wafv2, name, scope, fail_json_aws):
        self.wafv2 = wafv2
        self.name = name
        self.scope = scope
        self.fail_json_aws = fail_json_aws
        self.existing_set, self.id, self.locktoken, self.arn = self.get_set()

    def description(self):
        return self.existing_set.get('Description')

    def _format_set(self, ip_set):
        if ip_set is None:
            return None
        return camel_dict_to_snake_dict(self.existing_set, ignore_list=['tags'])

    def get(self):
        return self._format_set(self.existing_set)

    def remove(self):
        try:
            response = self.wafv2.delete_ip_set(Name=self.name, Scope=self.scope, Id=self.id, LockToken=self.locktoken)
        except (BotoCoreError, ClientError) as e:
            self.fail_json_aws(e, msg='Failed to remove wafv2 ip set.')
        return {}

    def create(self, description, ip_address_version, addresses, tags):
        req_obj = {'Name': self.name, 'Scope': self.scope, 'IPAddressVersion': ip_address_version, 'Addresses': addresses}
        if description:
            req_obj['Description'] = description
        if tags:
            req_obj['Tags'] = ansible_dict_to_boto3_tag_list(tags)
        try:
            response = self.wafv2.create_ip_set(**req_obj)
        except (BotoCoreError, ClientError) as e:
            self.fail_json_aws(e, msg='Failed to create wafv2 ip set.')
        self.existing_set, self.id, self.locktoken, self.arn = self.get_set()
        return self._format_set(self.existing_set)

    def update(self, description, addresses):
        req_obj = {'Name': self.name, 'Scope': self.scope, 'Id': self.id, 'Addresses': addresses, 'LockToken': self.locktoken}
        if description:
            req_obj['Description'] = description
        try:
            response = self.wafv2.update_ip_set(**req_obj)
        except (BotoCoreError, ClientError) as e:
            self.fail_json_aws(e, msg='Failed to update wafv2 ip set.')
        self.existing_set, self.id, self.locktoken, self.arn = self.get_set()
        return self._format_set(self.existing_set)

    def get_set(self):
        response = self.list()
        existing_set = None
        id = None
        arn = None
        locktoken = None
        for item in response.get('IPSets'):
            if item.get('Name') == self.name:
                id = item.get('Id')
                locktoken = item.get('LockToken')
                arn = item.get('ARN')
        if id:
            try:
                existing_set = self.wafv2.get_ip_set(Name=self.name, Scope=self.scope, Id=id).get('IPSet')
            except (BotoCoreError, ClientError) as e:
                self.fail_json_aws(e, msg='Failed to get wafv2 ip set.')
            tags = describe_wafv2_tags(self.wafv2, arn, self.fail_json_aws)
            existing_set['tags'] = tags
        return (existing_set, id, locktoken, arn)

    def list(self, Nextmarker=None):
        req_obj = {'Scope': self.scope, 'Limit': 100}
        if Nextmarker:
            req_obj['NextMarker'] = Nextmarker
        try:
            response = self.wafv2.list_ip_sets(**req_obj)
            if response.get('NextMarker'):
                response['IPSets'] += self.list(Nextmarker=response.get('NextMarker')).get('IPSets')
        except (BotoCoreError, ClientError) as e:
            self.fail_json_aws(e, msg='Failed to list wafv2 ip set.')
        return response