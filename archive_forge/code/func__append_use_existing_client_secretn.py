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
def _append_use_existing_client_secretn(action):
    if action['Type'] != 'authenticate-oidc':
        return action
    action['AuthenticateOidcConfig']['UseExistingClientSecret'] = True
    return action