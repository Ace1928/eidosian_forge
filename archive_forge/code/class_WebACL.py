from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
from ansible_collections.community.aws.plugins.module_utils.wafv2 import compare_priority_rules
from ansible_collections.community.aws.plugins.module_utils.wafv2 import describe_wafv2_tags
from ansible_collections.community.aws.plugins.module_utils.wafv2 import ensure_wafv2_tags
from ansible_collections.community.aws.plugins.module_utils.wafv2 import wafv2_list_web_acls
from ansible_collections.community.aws.plugins.module_utils.wafv2 import wafv2_snake_dict_to_camel_dict
class WebACL:

    def __init__(self, wafv2, name, scope, fail_json_aws):
        self.wafv2 = wafv2
        self.name = name
        self.scope = scope
        self.fail_json_aws = fail_json_aws
        self.existing_acl, self.id, self.locktoken = self.get_web_acl()

    def update(self, default_action, description, rules, sampled_requests, cloudwatch_metrics, metric_name, custom_response_bodies):
        req_obj = {'Name': self.name, 'Scope': self.scope, 'Id': self.id, 'DefaultAction': default_action, 'Rules': rules, 'VisibilityConfig': {'SampledRequestsEnabled': sampled_requests, 'CloudWatchMetricsEnabled': cloudwatch_metrics, 'MetricName': metric_name}, 'LockToken': self.locktoken}
        if description:
            req_obj['Description'] = description
        if custom_response_bodies:
            req_obj['CustomResponseBodies'] = custom_response_bodies
        try:
            response = self.wafv2.update_web_acl(**req_obj)
        except (BotoCoreError, ClientError) as e:
            self.fail_json_aws(e, msg='Failed to update wafv2 web acl.')
        self.existing_acl, self.id, self.locktoken = self.get_web_acl()
        return self.existing_acl

    def remove(self):
        try:
            response = self.wafv2.delete_web_acl(Name=self.name, Scope=self.scope, Id=self.id, LockToken=self.locktoken)
        except (BotoCoreError, ClientError) as e:
            self.fail_json_aws(e, msg='Failed to remove wafv2 web acl.')
        return response

    def get(self):
        if self.existing_acl:
            return self.existing_acl
        return None

    def get_web_acl(self):
        id = None
        locktoken = None
        arn = None
        existing_acl = None
        response = self.list()
        for item in response.get('WebACLs'):
            if item.get('Name') == self.name:
                id = item.get('Id')
                locktoken = item.get('LockToken')
                arn = item.get('ARN')
        if id:
            try:
                existing_acl = self.wafv2.get_web_acl(Name=self.name, Scope=self.scope, Id=id)
            except (BotoCoreError, ClientError) as e:
                self.fail_json_aws(e, msg='Failed to get wafv2 web acl.')
            tags = describe_wafv2_tags(self.wafv2, arn, self.fail_json_aws)
            existing_acl['tags'] = tags
        return (existing_acl, id, locktoken)

    def list(self):
        return wafv2_list_web_acls(self.wafv2, self.scope, self.fail_json_aws)

    def create(self, default_action, rules, sampled_requests, cloudwatch_metrics, metric_name, tags, description, custom_response_bodies):
        req_obj = {'Name': self.name, 'Scope': self.scope, 'DefaultAction': default_action, 'Rules': rules, 'VisibilityConfig': {'SampledRequestsEnabled': sampled_requests, 'CloudWatchMetricsEnabled': cloudwatch_metrics, 'MetricName': metric_name}}
        if custom_response_bodies:
            req_obj['CustomResponseBodies'] = custom_response_bodies
        if description:
            req_obj['Description'] = description
        if tags:
            req_obj['Tags'] = ansible_dict_to_boto3_tag_list(tags)
        try:
            response = self.wafv2.create_web_acl(**req_obj)
        except (BotoCoreError, ClientError) as e:
            self.fail_json_aws(e, msg='Failed to create wafv2 web acl.')
        self.existing_acl, self.id, self.locktoken = self.get_web_acl()
        return self.existing_acl