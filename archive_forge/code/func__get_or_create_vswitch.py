import logging
import os
import stat
from ray.autoscaler._private.aliyun.utils import AcsClient
def _get_or_create_vswitch(config):
    cli = _client(config)
    vswitches = cli.describe_v_switches(vpc_id=config['provider']['vpc_id'])
    if vswitches is not None and len(vswitches) > 0:
        config['provider']['v_switch_id'] = vswitches[0].get('VSwitchId')
        return
    v_switch_id = cli.create_v_switch(vpc_id=config['provider']['vpc_id'], zone_id=config['provider']['zone_id'], cidr_block=config['provider']['cidr_block'])
    if v_switch_id is not None:
        config['provider']['v_switch_id'] = v_switch_id