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
def _compare_rule(self, current_rule, new_rule):
    """

        :return:
        """
    modified_rule = {}
    if int(current_rule['Priority']) != int(new_rule['Priority']):
        modified_rule['Priority'] = new_rule['Priority']
    if len(current_rule['Actions']) == len(new_rule['Actions']):
        copy_new_rule = deepcopy(new_rule)
        current_actions_sorted = _sort_actions(current_rule['Actions'])
        new_actions_sorted = _sort_actions(copy_new_rule['Actions'])
        new_current_actions_sorted = [_append_use_existing_client_secretn(i) for i in current_actions_sorted]
        new_actions_sorted_no_secret = [_prune_secret(i) for i in new_actions_sorted]
        if [_prune_ForwardConfig(i) for i in new_current_actions_sorted] != [_prune_ForwardConfig(i) for i in new_actions_sorted_no_secret]:
            modified_rule['Actions'] = new_rule['Actions']
    else:
        modified_rule['Actions'] = new_rule['Actions']
    modified_conditions = []
    for condition in new_rule['Conditions']:
        if not self._compare_condition(current_rule['Conditions'], condition):
            modified_conditions.append(condition)
    if modified_conditions:
        modified_rule['Conditions'] = modified_conditions
    return modified_rule