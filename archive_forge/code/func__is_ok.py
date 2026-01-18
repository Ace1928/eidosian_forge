import time
import unittest
import boto.rds
from boto.vpc import VPCConnection
from boto.rds import RDSConnection
def _is_ok(subnet_group, vpc_id, description, subnets):
    if subnet_group.vpc_id != vpc_id:
        print('vpc_id is ', subnet_group.vpc_id, 'but should be ', vpc_id)
        return 0
    if subnet_group.description != description:
        print("description is '" + subnet_group.description + "' but should be '" + description + "'")
        return 0
    if set(subnet_group.subnet_ids) != set(subnets):
        subnets_are = ','.join(subnet_group.subnet_ids)
        should_be = ','.join(subnets)
        print('subnets are ' + subnets_are + ' but should be ' + should_be)
        return 0
    return 1