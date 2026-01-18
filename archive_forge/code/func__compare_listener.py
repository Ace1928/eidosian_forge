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
def _compare_listener(self, current_listener, new_listener):
    """
        Compare two listeners.

        :param current_listener:
        :param new_listener:
        :return:
        """
    modified_listener = {}
    if current_listener['Port'] != new_listener['Port']:
        modified_listener['Port'] = new_listener['Port']
    if current_listener['Protocol'] != new_listener['Protocol']:
        modified_listener['Protocol'] = new_listener['Protocol']
    if current_listener['Protocol'] == 'HTTPS' and new_listener['Protocol'] == 'HTTPS':
        if current_listener['SslPolicy'] != new_listener['SslPolicy']:
            modified_listener['SslPolicy'] = new_listener['SslPolicy']
        if current_listener['Certificates'][0]['CertificateArn'] != new_listener['Certificates'][0]['CertificateArn']:
            modified_listener['Certificates'] = []
            modified_listener['Certificates'].append({})
            modified_listener['Certificates'][0]['CertificateArn'] = new_listener['Certificates'][0]['CertificateArn']
    elif current_listener['Protocol'] != 'HTTPS' and new_listener['Protocol'] == 'HTTPS':
        modified_listener['SslPolicy'] = new_listener['SslPolicy']
        modified_listener['Certificates'] = []
        modified_listener['Certificates'].append({})
        modified_listener['Certificates'][0]['CertificateArn'] = new_listener['Certificates'][0]['CertificateArn']
    if len(current_listener['DefaultActions']) == len(new_listener['DefaultActions']):
        current_actions_sorted = _sort_actions(current_listener['DefaultActions'])
        new_actions_sorted = _sort_actions(new_listener['DefaultActions'])
        new_actions_sorted_no_secret = [_prune_secret(i) for i in new_actions_sorted]
        if [_prune_ForwardConfig(i) for i in current_actions_sorted] != [_prune_ForwardConfig(i) for i in new_actions_sorted_no_secret]:
            modified_listener['DefaultActions'] = new_listener['DefaultActions']
    else:
        modified_listener['DefaultActions'] = new_listener['DefaultActions']
    if modified_listener:
        return modified_listener
    else:
        return None