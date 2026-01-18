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
class ELBListener:

    def __init__(self, connection, module, listener, elb_arn):
        """

        :param connection:
        :param module:
        :param listener:
        :param elb_arn:
        """
        self.connection = connection
        self.module = module
        self.listener = listener
        self.elb_arn = elb_arn

    def add(self):
        try:
            if 'Rules' in self.listener:
                self.listener.pop('Rules')
            AWSRetry.jittered_backoff()(self.connection.create_listener)(LoadBalancerArn=self.elb_arn, **self.listener)
        except (BotoCoreError, ClientError) as e:
            self.module.fail_json_aws(e)

    def modify(self):
        try:
            if 'Rules' in self.listener:
                self.listener.pop('Rules')
            AWSRetry.jittered_backoff()(self.connection.modify_listener)(**self.listener)
        except (BotoCoreError, ClientError) as e:
            self.module.fail_json_aws(e)

    def delete(self):
        try:
            AWSRetry.jittered_backoff()(self.connection.delete_listener)(ListenerArn=self.listener)
        except (BotoCoreError, ClientError) as e:
            self.module.fail_json_aws(e)