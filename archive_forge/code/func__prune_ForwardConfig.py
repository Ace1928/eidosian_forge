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
def _prune_ForwardConfig(action):
    """
    Drops a redundant ForwardConfig where TargetGroupARN has already been set.
    (So we can perform comparisons)
    """
    if action.get('Type', '') != 'forward':
        return action
    if 'ForwardConfig' not in action:
        return action
    parent_arn = action.get('TargetGroupArn', None)
    arn = _simple_forward_config_arn(action['ForwardConfig'], parent_arn)
    if not arn:
        return action
    newAction = action.copy()
    del newAction['ForwardConfig']
    newAction['TargetGroupArn'] = arn
    return newAction