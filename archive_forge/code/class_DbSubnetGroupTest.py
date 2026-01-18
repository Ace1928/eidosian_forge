import time
import unittest
import boto.rds
from boto.vpc import VPCConnection
from boto.rds import RDSConnection
class DbSubnetGroupTest(unittest.TestCase):
    rds = True

    def test_db_subnet_group(self):
        vpc_api = VPCConnection()
        rds_api = RDSConnection()
        vpc = vpc_api.create_vpc('10.0.0.0/16')
        az_list = vpc_api.get_all_zones(filters={'state': 'available'})
        subnet = list()
        n = 0
        for az in az_list:
            try:
                subnet.append(vpc_api.create_subnet(vpc.id, '10.0.' + str(n) + '.0/24', availability_zone=az.name))
                n = n + 1
            except:
                pass
        grp_name = 'db_subnet_group' + str(int(time.time()))
        subnet_group = rds_api.create_db_subnet_group(grp_name, grp_name, [subnet[0].id, subnet[1].id])
        if not _is_ok(subnet_group, vpc.id, grp_name, [subnet[0].id, subnet[1].id]):
            raise Exception('create_db_subnet_group returned bad values')
        rds_api.modify_db_subnet_group(grp_name, description='new description')
        subnet_grps = rds_api.get_all_db_subnet_groups(name=grp_name)
        if not _is_ok(subnet_grps[0], vpc.id, 'new description', [subnet[0].id, subnet[1].id]):
            raise Exception('modifying the subnet group desciption returned bad values')
        rds_api.modify_db_subnet_group(grp_name, subnet_ids=[subnet[1].id, subnet[2].id])
        subnet_grps = rds_api.get_all_db_subnet_groups(name=grp_name)
        if not _is_ok(subnet_grps[0], vpc.id, 'new description', [subnet[1].id, subnet[2].id]):
            raise Exception('modifying the subnet group subnets returned bad values')
        rds_api.delete_db_subnet_group(subnet_group.name)
        try:
            rds_api.get_all_db_subnet_groups(name=grp_name)
            raise Exception(subnet_group.name + ' still accessible after delete_db_subnet_group')
        except:
            pass
        while n > 0:
            n = n - 1
            vpc_api.delete_subnet(subnet[n].id)
        vpc_api.delete_vpc(vpc.id)