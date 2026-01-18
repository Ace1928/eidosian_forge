from .botocore import is_boto3_error_code
from .retries import AWSRetry
@AWSRetry.jittered_backoff()
def _get_elb(connection, module, elb_name):
    """
    Get an ELB based on name using AWSRetry. If not found, return None.

    :param connection: AWS boto3 elbv2 connection
    :param module: Ansible module
    :param elb_name: Name of load balancer to get
    :return: boto3 ELB dict or None if not found
    """
    try:
        load_balancer_paginator = connection.get_paginator('describe_load_balancers')
        return load_balancer_paginator.paginate(Names=[elb_name]).build_full_result()['LoadBalancers'][0]
    except is_boto3_error_code('LoadBalancerNotFound'):
        return None