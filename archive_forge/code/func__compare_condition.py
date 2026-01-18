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
def _compare_condition(self, current_conditions, condition):
    """

        :param current_conditions:
        :param condition:
        :return:
        """
    condition_found = False
    for current_condition in current_conditions:
        if current_condition.get('HostHeaderConfig') and condition.get('HostHeaderConfig'):
            if current_condition['Field'] == condition['Field'] and sorted(current_condition['HostHeaderConfig']['Values']) == sorted(condition['HostHeaderConfig']['Values']):
                condition_found = True
                break
        elif current_condition.get('HttpHeaderConfig'):
            if current_condition['Field'] == condition['Field'] and sorted(current_condition['HttpHeaderConfig']['Values']) == sorted(condition['HttpHeaderConfig']['Values']) and (current_condition['HttpHeaderConfig']['HttpHeaderName'] == condition['HttpHeaderConfig']['HttpHeaderName']):
                condition_found = True
                break
        elif current_condition.get('HttpRequestMethodConfig'):
            if current_condition['Field'] == condition['Field'] and sorted(current_condition['HttpRequestMethodConfig']['Values']) == sorted(condition['HttpRequestMethodConfig']['Values']):
                condition_found = True
                break
        elif current_condition.get('PathPatternConfig') and condition.get('PathPatternConfig'):
            if current_condition['Field'] == condition['Field'] and sorted(current_condition['PathPatternConfig']['Values']) == sorted(condition['PathPatternConfig']['Values']):
                condition_found = True
                break
        elif current_condition.get('QueryStringConfig'):
            if current_condition['Field'] == condition['Field'] and current_condition['QueryStringConfig']['Values'] == condition['QueryStringConfig']['Values']:
                condition_found = True
                break
        elif current_condition.get('SourceIpConfig'):
            if current_condition['Field'] == condition['Field'] and sorted(current_condition['SourceIpConfig']['Values']) == sorted(condition['SourceIpConfig']['Values']):
                condition_found = True
                break
        elif current_condition['Field'] == condition['Field'] and sorted(current_condition['Values']) == sorted(condition['Values']):
            condition_found = True
            break
    return condition_found