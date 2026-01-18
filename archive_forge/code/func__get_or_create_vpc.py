import logging
import os
import stat
from ray.autoscaler._private.aliyun.utils import AcsClient
def _get_or_create_vpc(config):
    cli = _client(config)
    vpcs = cli.describe_vpcs()
    if vpcs is not None and len(vpcs) > 0:
        config['provider']['vpc_id'] = vpcs[0].get('VpcId')
        return
    vpc_id = cli.create_vpc()
    if vpc_id is not None:
        config['provider']['vpc_id'] = vpc_id