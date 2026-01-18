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
def _simple_forward_config_arn(config, parent_arn):
    config = deepcopy(config)
    stickiness = config.pop('TargetGroupStickinessConfig', {'Enabled': False})
    if stickiness != {'Enabled': False}:
        return False
    target_groups = config.pop('TargetGroups', [])
    if config:
        return False
    if len(target_groups) > 1:
        return False
    if not target_groups:
        return parent_arn or False
    target_group = target_groups[0]
    target_group.pop('Weight', None)
    target_group_arn = target_group.pop('TargetGroupArn', None)
    if target_group:
        return False
    if not (target_group_arn or parent_arn):
        return False
    if not parent_arn:
        return target_group_arn
    if not target_group_arn:
        return parent_arn
    if parent_arn != target_group_arn:
        return False
    return target_group_arn