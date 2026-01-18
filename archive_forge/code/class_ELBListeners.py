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
class ELBListeners:

    def __init__(self, connection, module, elb_arn):
        self.connection = connection
        self.module = module
        self.elb_arn = elb_arn
        listeners = module.params.get('listeners')
        if listeners is not None:
            listeners = [dict(((x, listener_dict[x]) for x in listener_dict if listener_dict[x] is not None)) for listener_dict in listeners]
        self.listeners = self._ensure_listeners_default_action_has_arn(listeners)
        self.current_listeners = self._get_elb_listeners()
        self.purge_listeners = module.params.get('purge_listeners')
        self.changed = False

    def update(self):
        """
        Update the listeners for the ELB

        :return:
        """
        self.current_listeners = self._get_elb_listeners()

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

    def _ensure_listeners_default_action_has_arn(self, listeners):
        """
        If a listener DefaultAction has been passed with a Target Group Name instead of ARN, lookup the ARN and
        replace the name.

        :param listeners: a list of listener dicts
        :return: the same list of dicts ensuring that each listener DefaultActions dict has TargetGroupArn key. If a TargetGroupName key exists, it is removed.
        """
        if not listeners:
            listeners = []
        fixed_listeners = []
        for listener in listeners:
            fixed_actions = []
            for action in listener['DefaultActions']:
                if 'TargetGroupName' in action:
                    action['TargetGroupArn'] = convert_tg_name_to_arn(self.connection, self.module, action['TargetGroupName'])
                    del action['TargetGroupName']
                fixed_actions.append(action)
            listener['DefaultActions'] = fixed_actions
            fixed_listeners.append(listener)
        return fixed_listeners

    def compare_listeners(self):
        """

        :return:
        """
        listeners_to_modify = []
        listeners_to_delete = []
        listeners_to_add = deepcopy(self.listeners)
        for current_listener in self.current_listeners:
            current_listener_passed_to_module = False
            for new_listener in self.listeners[:]:
                new_listener['Port'] = int(new_listener['Port'])
                if current_listener['Port'] == new_listener['Port']:
                    current_listener_passed_to_module = True
                    listeners_to_add.remove(new_listener)
                    modified_listener = self._compare_listener(current_listener, new_listener)
                    if modified_listener:
                        modified_listener['Port'] = current_listener['Port']
                        modified_listener['ListenerArn'] = current_listener['ListenerArn']
                        listeners_to_modify.append(modified_listener)
                    break
            if not current_listener_passed_to_module and self.purge_listeners:
                listeners_to_delete.append(current_listener['ListenerArn'])
        return (listeners_to_add, listeners_to_modify, listeners_to_delete)

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