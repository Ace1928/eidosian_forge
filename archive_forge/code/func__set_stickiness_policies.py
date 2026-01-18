from ansible.module_utils.common.dict_transformations import camel_dict_to_snake_dict
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.ec2 import get_ec2_security_group_ids_from_names
from ansible_collections.amazon.aws.plugins.module_utils.modules import AnsibleAWSModule
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.amazon.aws.plugins.module_utils.tagging import ansible_dict_to_boto3_tag_list
from ansible_collections.amazon.aws.plugins.module_utils.tagging import boto3_tag_list_to_ansible_dict
from ansible_collections.amazon.aws.plugins.module_utils.tagging import compare_aws_tags
from ansible_collections.amazon.aws.plugins.module_utils.transformation import ansible_dict_to_boto3_filter_list
from ansible_collections.amazon.aws.plugins.module_utils.transformation import scrub_none_parameters
from ansible_collections.amazon.aws.plugins.module_utils.waiters import get_waiter
def _set_stickiness_policies(self):
    if self.stickiness is None:
        return False
    self._update_descriptions()
    if not self.stickiness['enabled']:
        return self._purge_stickiness_policies()
    if self.stickiness['type'] == 'loadbalancer':
        policy_name = self._policy_name('LBCookieStickinessPolicyType')
        expiration = self.stickiness.get('expiration')
        if not expiration:
            expiration = 0
        policy_description = dict(PolicyName=policy_name, CookieExpirationPeriod=expiration)
        existing_policies = self._get_lb_stickness_policy_map()
        add_method = self.client.create_lb_cookie_stickiness_policy
    elif self.stickiness['type'] == 'application':
        policy_name = self._policy_name('AppCookieStickinessPolicyType')
        policy_description = dict(PolicyName=policy_name, CookieName=self.stickiness.get('cookie', 0))
        existing_policies = self._get_app_stickness_policy_map()
        add_method = self.client.create_app_cookie_stickiness_policy
    else:
        self.module.fail_json(msg=f'Unknown stickiness policy {self.stickiness['type']}')
    changed = False
    if policy_name in existing_policies:
        if existing_policies[policy_name] != policy_description:
            changed |= self._purge_stickiness_policies()
    if changed:
        self._update_descriptions()
    changed |= self._set_stickiness_policy(method=add_method, description=policy_description, existing_policies=existing_policies)
    listeners = self.elb['ListenerDescriptions']
    for listener in listeners:
        changed |= self._set_lb_stickiness_policy(listener=listener, policy=policy_name)
    return changed