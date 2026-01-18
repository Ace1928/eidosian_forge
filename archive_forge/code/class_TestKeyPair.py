import contextlib
import datetime
from unittest import mock
import uuid
import warnings
from openstack.block_storage.v3 import volume
from openstack.compute.v2 import _proxy
from openstack.compute.v2 import aggregate
from openstack.compute.v2 import availability_zone as az
from openstack.compute.v2 import extension
from openstack.compute.v2 import flavor
from openstack.compute.v2 import hypervisor
from openstack.compute.v2 import image
from openstack.compute.v2 import keypair
from openstack.compute.v2 import migration
from openstack.compute.v2 import quota_set
from openstack.compute.v2 import server
from openstack.compute.v2 import server_action
from openstack.compute.v2 import server_group
from openstack.compute.v2 import server_interface
from openstack.compute.v2 import server_ip
from openstack.compute.v2 import server_migration
from openstack.compute.v2 import server_remote_console
from openstack.compute.v2 import service
from openstack.compute.v2 import usage
from openstack.compute.v2 import volume_attachment
from openstack import resource
from openstack.tests.unit import test_proxy_base
from openstack import warnings as os_warnings
class TestKeyPair(TestComputeProxy):

    def test_keypair_create(self):
        self.verify_create(self.proxy.create_keypair, keypair.Keypair)

    def test_keypair_delete(self):
        self.verify_delete(self.proxy.delete_keypair, keypair.Keypair, False)

    def test_keypair_delete_ignore(self):
        self.verify_delete(self.proxy.delete_keypair, keypair.Keypair, True)

    def test_keypair_delete_user_id(self):
        self.verify_delete(self.proxy.delete_keypair, keypair.Keypair, True, method_kwargs={'user_id': 'fake_user'}, expected_kwargs={'user_id': 'fake_user'})

    def test_keypair_find(self):
        self.verify_find(self.proxy.find_keypair, keypair.Keypair)

    def test_keypair_find_user_id(self):
        self.verify_find(self.proxy.find_keypair, keypair.Keypair, method_kwargs={'user_id': 'fake_user'}, expected_kwargs={'user_id': 'fake_user'})

    def test_keypair_get(self):
        self.verify_get(self.proxy.get_keypair, keypair.Keypair)

    def test_keypair_get_user_id(self):
        self.verify_get(self.proxy.get_keypair, keypair.Keypair, method_kwargs={'user_id': 'fake_user'}, expected_kwargs={'user_id': 'fake_user'})

    def test_keypairs(self):
        self.verify_list(self.proxy.keypairs, keypair.Keypair)

    def test_keypairs_user_id(self):
        self.verify_list(self.proxy.keypairs, keypair.Keypair, method_kwargs={'user_id': 'fake_user'}, expected_kwargs={'user_id': 'fake_user'})