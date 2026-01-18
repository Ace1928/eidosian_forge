import traceback
from copy import deepcopy
from .ec2 import get_ec2_security_group_ids_from_names
from .elb_utils import convert_tg_name_to_arn
from .elb_utils import get_elb
from .elb_utils import get_elb_listener
from .retries import AWSRetry
from .tagging import ansible_dict_to_boto3_tag_list
from .tagging import boto3_tag_list_to_ansible_dict
from .waiters import get_waiter
def _ensure_rules_action_has_arn(self, rules):
    """
        If a rule Action has been passed with a Target Group Name instead of ARN, lookup the ARN and
        replace the name.

        :param rules: a list of rule dicts
        :return: the same list of dicts ensuring that each rule Actions dict has TargetGroupArn key. If a TargetGroupName key exists, it is removed.
        """
    fixed_rules = []
    for rule in rules:
        fixed_actions = []
        for action in rule['Actions']:
            if 'TargetGroupName' in action:
                action['TargetGroupArn'] = convert_tg_name_to_arn(self.connection, self.module, action['TargetGroupName'])
                del action['TargetGroupName']
            fixed_actions.append(action)
        rule['Actions'] = fixed_actions
        fixed_rules.append(rule)
    return fixed_rules