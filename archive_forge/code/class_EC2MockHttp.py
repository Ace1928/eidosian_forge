import os
import sys
import base64
from datetime import datetime
from collections import OrderedDict
from libcloud.test import MockHttp, LibcloudTestCase, unittest
from libcloud.utils.py3 import b, httplib, parse_qs
from libcloud.compute.base import (
from libcloud.test.compute import TestCaseMixin
from libcloud.test.secrets import EC2_PARAMS
from libcloud.compute.types import (
from libcloud.utils.iso8601 import UTC
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.ec2 import (
class EC2MockHttp(MockHttp, unittest.TestCase):
    fixtures = ComputeFileFixtures('ec2')

    def _DescribeInstances(self, method, url, body, headers):
        body = self.fixtures.load('describe_instances.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _DescribeReservedInstances(self, method, url, body, headers):
        body = self.fixtures.load('describe_reserved_instances.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _DescribeAvailabilityZones(self, method, url, body, headers):
        body = self.fixtures.load('describe_availability_zones.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _RebootInstances(self, method, url, body, headers):
        body = self.fixtures.load('reboot_instances.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _StartInstances(self, method, url, body, headers):
        body = self.fixtures.load('start_instances.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _StopInstances(self, method, url, body, headers):
        body = self.fixtures.load('stop_instances.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _DescribeSecurityGroups(self, method, url, body, headers):
        body = self.fixtures.load('describe_security_groups.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _DeleteSecurityGroup(self, method, url, body, headers):
        body = self.fixtures.load('delete_security_group.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _AuthorizeSecurityGroupIngress(self, method, url, body, headers):
        body = self.fixtures.load('authorize_security_group_ingress.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _DescribeImages(self, method, url, body, headers):
        body = self.fixtures.load('describe_images.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _RegisterImages(self, method, url, body, headers):
        body = self.fixtures.load('register_image.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _ImportSnapshot(self, method, url, body, headers):
        body = self.fixtures.load('import_snapshot.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _DescribeImportSnapshotTasks(self, method, url, body, headers):
        body = self.fixtures.load('describe_import_snapshot_tasks.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _timeout_DescribeImportSnapshotTasks(self, method, url, body, headers):
        body = self.fixtures.load('describe_import_snapshot_tasks_active.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _ex_imageids_DescribeImages(self, method, url, body, headers):
        body = self.fixtures.load('describe_images_ex_imageids.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _RunInstances(self, method, url, body, headers):
        body = self.fixtures.load('run_instances.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _ex_user_data_RunInstances(self, method, url, body, headers):
        if url.startswith('/'):
            url = url[1:]
        if url.startswith('?'):
            url = url[1:]
        params = parse_qs(url)
        self.assertTrue('UserData' in params, 'UserData not in params, actual params: %s' % str(params))
        user_data = base64.b64decode(b(params['UserData'][0])).decode('utf-8')
        self.assertEqual(user_data, 'foo\nbar\x0coo')
        body = self.fixtures.load('run_instances.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _create_ex_assign_public_ip_RunInstances(self, method, url, body, headers):
        self.assertUrlContainsQueryParams(url, {'NetworkInterface.1.AssociatePublicIpAddress': 'true', 'NetworkInterface.1.DeleteOnTermination': 'true', 'NetworkInterface.1.DeviceIndex': '0', 'NetworkInterface.1.SubnetId': 'subnet-11111111', 'NetworkInterface.1.SecurityGroupId.1': 'sg-11111111'})
        body = self.fixtures.load('run_instances_with_subnet_and_security_group.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _create_ex_terminate_on_shutdown_RunInstances(self, method, url, body, headers):
        self.assertUrlContainsQueryParams(url, {'InstanceInitiatedShutdownBehavior': 'terminate'})
        body = self.fixtures.load('run_instances.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _ex_security_groups_RunInstances(self, method, url, body, headers):
        self.assertUrlContainsQueryParams(url, {'SecurityGroup.1': 'group1'})
        self.assertUrlContainsQueryParams(url, {'SecurityGroup.2': 'group2'})
        body = self.fixtures.load('run_instances.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _ex_security_group_ids_RunInstances(self, method, url, body, headers):
        self.assertUrlContainsQueryParams(url, {'SecurityGroupId.1': 'sg-1aa11a1a'})
        self.assertUrlContainsQueryParams(url, {'SecurityGroupId.2': 'sg-2bb22b2b'})
        body = self.fixtures.load('run_instances.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _create_ex_blockdevicemappings_RunInstances(self, method, url, body, headers):
        expected_params = {'BlockDeviceMapping.1.DeviceName': '/dev/sda1', 'BlockDeviceMapping.1.Ebs.VolumeSize': '10', 'BlockDeviceMapping.2.DeviceName': '/dev/sdb', 'BlockDeviceMapping.2.VirtualName': 'ephemeral0', 'BlockDeviceMapping.3.DeviceName': '/dev/sdc', 'BlockDeviceMapping.3.VirtualName': 'ephemeral1'}
        self.assertUrlContainsQueryParams(url, expected_params)
        body = self.fixtures.load('run_instances.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _idempotent_RunInstances(self, method, url, body, headers):
        body = self.fixtures.load('run_instances_idem.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _idempotent_mismatch_RunInstances(self, method, url, body, headers):
        body = self.fixtures.load('run_instances_idem_mismatch.xml')
        return (httplib.BAD_REQUEST, body, {}, httplib.responses[httplib.BAD_REQUEST])

    def _ex_iam_profile_RunInstances(self, method, url, body, headers):
        body = self.fixtures.load('run_instances_iam_profile.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _ex_spot_RunInstances(self, method, url, body, headers):
        body = self.fixtures.load('run_instances_spot.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _TerminateInstances(self, method, url, body, headers):
        body = self.fixtures.load('terminate_instances.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _DescribeKeyPairs(self, method, url, body, headers):
        body = self.fixtures.load('describe_key_pairs.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _get_one_DescribeKeyPairs(self, method, url, body, headers):
        self.assertUrlContainsQueryParams(url, {'KeyName': 'gsg-keypair'})
        body = self.fixtures.load('describe_key_pairs.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _doesnt_exist_DescribeKeyPairs(self, method, url, body, headers):
        body = self.fixtures.load('describe_key_pairs_doesnt_exist.xml')
        return (httplib.BAD_REQUEST, body, {}, httplib.responses[httplib.BAD_REQUEST])

    def _CreateKeyPair(self, method, url, body, headers):
        body = self.fixtures.load('create_key_pair.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _ImportKeyPair(self, method, url, body, headers):
        body = self.fixtures.load('import_key_pair.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _DescribeTags(self, method, url, body, headers):
        body = self.fixtures.load('describe_tags.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _CreateTags(self, method, url, body, headers):
        body = self.fixtures.load('create_tags.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _DeleteTags(self, method, url, body, headers):
        body = self.fixtures.load('delete_tags.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _DescribeAddresses(self, method, url, body, headers):
        body = self.fixtures.load('describe_addresses_multi.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _AllocateAddress(self, method, url, body, headers):
        body = self.fixtures.load('allocate_address.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _vpc_AllocateAddress(self, method, url, body, headers):
        body = self.fixtures.load('allocate_vpc_address.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _AssociateAddress(self, method, url, body, headers):
        body = self.fixtures.load('associate_address.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _vpc_AssociateAddress(self, method, url, body, headers):
        body = self.fixtures.load('associate_vpc_address.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _DisassociateAddress(self, method, url, body, headers):
        body = self.fixtures.load('disassociate_address.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _ReleaseAddress(self, method, url, body, headers):
        body = self.fixtures.load('release_address.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _all_addresses_DescribeAddresses(self, method, url, body, headers):
        body = self.fixtures.load('describe_addresses_all.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _WITH_TAGS_DescribeAddresses(self, method, url, body, headers):
        body = self.fixtures.load('describe_addresses_multi.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _ModifyInstanceAttribute(self, method, url, body, headers):
        body = self.fixtures.load('modify_instance_attribute.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _ModifySnapshotAttribute(self, method, url, body, headers):
        body = self.fixtures.load('modify_snapshot_attribute.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _CreateVolume(self, method, url, body, headers):
        if 'KmsKeyId=' in url:
            assert 'Encrypted=1' in url, 'If a KmsKeyId is specified, the Encrypted flag must also be set.'
        if 'Encrypted=1' in url:
            body = self.fixtures.load('create_encrypted_volume.xml')
        else:
            body = self.fixtures.load('create_volume.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _DeleteVolume(self, method, url, body, headers):
        body = self.fixtures.load('delete_volume.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _AttachVolume(self, method, url, body, headers):
        body = self.fixtures.load('attach_volume.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _DetachVolume(self, method, url, body, headers):
        body = self.fixtures.load('detach_volume.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _DescribeVolumes(self, method, url, body, headers):
        body = self.fixtures.load('describe_volumes.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _filters_nodes_DescribeVolumes(self, method, url, body, headers):
        expected_params = {'Filter.1.Name': 'attachment.instance-id', 'Filter.1.Value.1': 'i-d334b4b3'}
        self.assertUrlContainsQueryParams(url, expected_params)
        body = self.fixtures.load('describe_volumes.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _filters_status_DescribeVolumes(self, method, url, body, headers):
        expected_params = {'Filter.1.Name': 'status', 'Filter.1.Value.1': 'available'}
        self.assertUrlContainsQueryParams(url, expected_params)
        body = self.fixtures.load('describe_volumes.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _CreateSnapshot(self, method, url, body, headers):
        body = self.fixtures.load('create_snapshot.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _DescribeSnapshots(self, method, url, body, headers):
        body = self.fixtures.load('describe_snapshots.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _DeleteSnapshot(self, method, url, body, headers):
        body = self.fixtures.load('delete_snapshot.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _CopyImage(self, method, url, body, headers):
        body = self.fixtures.load('copy_image.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _CreateImage(self, method, url, body, headers):
        body = self.fixtures.load('create_image.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _DeregisterImage(self, method, url, body, headers):
        body = self.fixtures.load('deregister_image.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _DeleteKeyPair(self, method, url, body, headers):
        self.assertUrlContainsQueryParams(url, {'KeyName': 'gsg-keypair'})
        body = self.fixtures.load('delete_key_pair.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _ModifyImageAttribute(self, method, url, body, headers):
        body = self.fixtures.load('modify_image_attribute.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _DescribeAccountAttributes(self, method, url, body, headers):
        body = self.fixtures.load('describe_account_attributes.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _CreateSecurityGroup(self, method, url, body, headers):
        body = self.fixtures.load('create_security_group.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _DescribeVpcs(self, method, url, body, headers):
        body = self.fixtures.load('describe_vpcs.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _network_ids_DescribeVpcs(self, method, url, body, headers):
        expected_params = {'VpcId.1': 'vpc-532335e1'}
        self.assertUrlContainsQueryParams(url, expected_params)
        body = self.fixtures.load('describe_vpcs.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _filters_DescribeVpcs(self, method, url, body, headers):
        expected_params = {'Filter.1.Name': 'dhcp-options-id', 'Filter.1.Value.1': 'dopt-7eded312', 'Filter.2.Name': 'cidr', 'Filter.2.Value.1': '192.168.51.0/24'}
        self.assertUrlContainsQueryParams(url, expected_params)
        body = self.fixtures.load('describe_vpcs.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _CreateVpc(self, method, url, body, headers):
        body = self.fixtures.load('create_vpc.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _DeleteVpc(self, method, url, body, headers):
        body = self.fixtures.load('delete_vpc.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _DescribeSubnets(self, method, url, body, headers):
        body = self.fixtures.load('describe_subnets.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _CreateSubnet(self, method, url, body, headers):
        body = self.fixtures.load('create_subnet.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _ModifySubnetAttribute(self, method, url, body, headers):
        body = self.fixtures.load('modify_subnet_attribute.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _DeleteSubnet(self, method, url, body, headers):
        body = self.fixtures.load('delete_subnet.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _GetConsoleOutput(self, method, url, body, headers):
        body = self.fixtures.load('get_console_output.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _DescribeNetworkInterfaces(self, method, url, body, headers):
        body = self.fixtures.load('describe_network_interfaces.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _CreateNetworkInterface(self, method, url, body, headers):
        body = self.fixtures.load('create_network_interface.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _DeleteNetworkInterface(self, method, url, body, headers):
        body = self.fixtures.load('delete_network_interface.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _AttachNetworkInterface(self, method, url, body, headers):
        body = self.fixtures.load('attach_network_interface.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _DetachNetworkInterface(self, method, url, body, headers):
        body = self.fixtures.load('detach_network_interface.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _DescribeInternetGateways(self, method, url, body, headers):
        body = self.fixtures.load('describe_internet_gateways.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _CreateInternetGateway(self, method, url, body, headers):
        body = self.fixtures.load('create_internet_gateway.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _DeleteInternetGateway(self, method, url, body, headers):
        body = self.fixtures.load('delete_internet_gateway.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _AttachInternetGateway(self, method, url, body, headers):
        body = self.fixtures.load('attach_internet_gateway.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _DetachInternetGateway(self, method, url, body, headers):
        body = self.fixtures.load('detach_internet_gateway.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _CreatePlacementGroup(self, method, url, body, headers):
        body = self.fixtures.load('create_placement_groups.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _DeletePlacementGroup(self, method, url, body, headers):
        body = self.fixtures.load('delete_placement_groups.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _DescribePlacementGroups(self, method, url, body, headers):
        body = self.fixtures.load('describe_placement_groups.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _ModifyVolume(self, method, url, body, headers):
        body = self.fixtures.load('modify_volume.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])

    def _DescribeVolumesModifications(self, method, url, body, headers):
        body = self.fixtures.load('describe_volumes_modifications.xml')
        return (httplib.OK, body, {}, httplib.responses[httplib.OK])