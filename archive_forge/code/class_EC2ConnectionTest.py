import unittest
import time
import telnetlib
import socket
from nose.plugins.attrib import attr
from boto.ec2.connection import EC2Connection
from boto.exception import EC2ResponseError
import boto.ec2
class EC2ConnectionTest(unittest.TestCase):
    ec2 = True

    @attr('notdefault')
    def test_launch_permissions(self):
        user_id = '963068290131'
        print('--- running EC2Connection tests ---')
        c = EC2Connection()
        rs = c.get_all_images(owners=[user_id])
        assert len(rs) > 0
        image = rs[0]
        status = image.set_launch_permissions(group_names=['all'])
        assert status
        d = image.get_launch_permissions()
        assert 'groups' in d
        assert len(d['groups']) > 0
        status = image.remove_launch_permissions(group_names=['all'])
        assert status
        time.sleep(10)
        d = image.get_launch_permissions()
        assert 'groups' not in d

    def test_1_basic(self):
        c = EC2Connection()
        group1_name = 'test-%d' % int(time.time())
        group_desc = 'This is a security group created during unit testing'
        group1 = c.create_security_group(group1_name, group_desc)
        time.sleep(2)
        group2_name = 'test-%d' % int(time.time())
        group_desc = 'This is a security group created during unit testing'
        group2 = c.create_security_group(group2_name, group_desc)
        rs = c.get_all_security_groups()
        found = False
        for g in rs:
            if g.name == group1_name:
                found = True
        assert found
        rs = c.get_all_security_groups([group1_name])
        assert len(rs) == 1
        status = c.authorize_security_group(group1.name, group2.name, group2.owner_id)
        assert status
        status = c.revoke_security_group(group1.name, group2.name, group2.owner_id)
        assert status
        status = c.authorize_security_group(group1.name, group2.name, group2.owner_id, 'tcp', 22, 22)
        assert status
        status = c.revoke_security_group(group1.name, group2.name, group2.owner_id, 'tcp', 22, 22)
        assert status
        status = c.delete_security_group(group2_name)
        rs = c.get_all_security_groups()
        found = False
        for g in rs:
            if g.name == group2_name:
                found = True
        assert not found
        group = group1
        rs = c.get_all_images()
        img_loc = 'ec2-public-images/fedora-core4-apache.manifest.xml'
        for image in rs:
            if image.location == img_loc:
                break
        reservation = image.run(security_groups=[group.name])
        instance = reservation.instances[0]
        while instance.state != 'running':
            print('\tinstance is %s' % instance.state)
            time.sleep(30)
            instance.update()
        t = telnetlib.Telnet()
        try:
            t.open(instance.dns_name, 80)
        except socket.error:
            pass
        group.authorize('tcp', 80, 80, '0.0.0.0/0')
        t.open(instance.dns_name, 80)
        t.close()
        group.revoke('tcp', 80, 80, '0.0.0.0/0')
        try:
            t.open(instance.dns_name, 80)
        except socket.error:
            pass
        instance.terminate()
        assert instance.state == 'shutting-down'
        assert instance.state_code == 32
        assert instance.previous_state == 'running'
        assert instance.previous_state_code == 16
        key_name = 'test-%d' % int(time.time())
        status = c.create_key_pair(key_name)
        assert status
        rs = c.get_all_key_pairs()
        found = False
        for k in rs:
            if k.name == key_name:
                found = True
        assert found
        rs = c.get_all_key_pairs([key_name])
        assert len(rs) == 1
        key_pair = rs[0]
        status = c.delete_key_pair(key_name)
        rs = c.get_all_key_pairs()
        found = False
        for k in rs:
            if k.name == key_name:
                found = True
        assert not found
        demo_paid_ami_id = 'ami-bd9d78d4'
        demo_paid_ami_product_code = 'A79EC0DB'
        l = c.get_all_images([demo_paid_ami_id])
        assert len(l) == 1
        assert len(l[0].product_codes) == 1
        assert l[0].product_codes[0] == demo_paid_ami_product_code
        print('--- tests completed ---')

    def test_dry_run(self):
        c = EC2Connection()
        dry_run_msg = 'Request would have succeeded, but DryRun flag is set.'
        try:
            rs = c.get_all_images(dry_run=True)
            self.fail('Should have gotten an exception')
        except EC2ResponseError as e:
            self.assertTrue(dry_run_msg in str(e))
        try:
            rs = c.run_instances(image_id='ami-a0cd60c9', instance_type='m1.small', dry_run=True)
            self.fail('Should have gotten an exception')
        except EC2ResponseError as e:
            self.assertTrue(dry_run_msg in str(e))
        rs = c.run_instances(image_id='ami-a0cd60c9', instance_type='m1.small')
        time.sleep(120)
        try:
            rs = c.stop_instances(instance_ids=[rs.instances[0].id], dry_run=True)
            self.fail('Should have gotten an exception')
        except EC2ResponseError as e:
            self.assertTrue(dry_run_msg in str(e))
        try:
            rs = c.terminate_instances(instance_ids=[rs.instances[0].id], dry_run=True)
            self.fail('Should have gotten an exception')
        except EC2ResponseError as e:
            self.assertTrue(dry_run_msg in str(e))
        rs.instances[0].terminate()

    def test_can_get_all_instances_sigv4(self):
        connection = boto.ec2.connect_to_region('eu-central-1')
        self.assertTrue(isinstance(connection.get_all_instances(), list))