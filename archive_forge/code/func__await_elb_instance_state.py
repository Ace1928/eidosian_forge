from ansible_collections.amazon.aws.plugins.module_utils.botocore import is_boto3_error_code
from ansible_collections.amazon.aws.plugins.module_utils.retries import AWSRetry
from ansible_collections.community.aws.plugins.module_utils.modules import AnsibleCommunityAWSModule as AnsibleAWSModule
def _await_elb_instance_state(self, lb, awaited_state, timeout):
    """Wait for an ELB to change state"""
    if self.module.check_mode:
        return
    initial_state = self._get_instance_health(lb)
    if awaited_state == initial_state:
        return
    if awaited_state == 'InService':
        waiter = self.client_elb.get_waiter('instance_in_service')
    elif awaited_state == 'Deregistered':
        waiter = self.client_elb.get_waiter('instance_deregistered')
    elif awaited_state == 'OutOfService':
        waiter = self.client_elb.get_waiter('instance_deregistered')
    else:
        self.module.fail_json(msg='Could not wait for unknown state', awaited_state=awaited_state)
    try:
        waiter.wait(LoadBalancerName=lb['LoadBalancerName'], Instances=[{'InstanceId': self.instance_id}], WaiterConfig={'Delay': 1, 'MaxAttempts': timeout})
    except botocore.exceptions.WaiterError as e:
        self.module.fail_json_aws(e, msg='Timeout waiting for instance to reach desired state', awaited_state=awaited_state)
    except (botocore.exceptions.ClientError, botocore.exceptions.BotoCoreError) as e:
        self.module.fail_json_aws(e, msg='Error while waiting for instance to reach desired state', awaited_state=awaited_state)
    return