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
def _get_elb_listeners(self):
    """
        Get ELB listeners

        :return:
        """
    try:
        listener_paginator = self.connection.get_paginator('describe_listeners')
        return AWSRetry.jittered_backoff()(listener_paginator.paginate)(LoadBalancerArn=self.elb_arn).build_full_result()['Listeners']
    except (BotoCoreError, ClientError) as e:
        self.module.fail_json_aws(e)