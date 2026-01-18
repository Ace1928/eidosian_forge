import sys
import unittest
from libcloud.common.base import Connection, ConnectionKey, ConnectionUserAndKey
from libcloud.common.types import LibcloudError
from libcloud.compute.base import (
from libcloud.compute.types import StorageVolumeState
class TestValidateAuth(unittest.TestCase):

    def test_get_auth_ssh(self):
        n = NodeDriver('foo')
        n.features = {'create_node': ['ssh_key']}
        auth = NodeAuthSSHKey('pubkey...')
        self.assertEqual(auth, n._get_and_check_auth(auth))

    def test_get_auth_ssh_but_given_password(self):
        n = NodeDriver('foo')
        n.features = {'create_node': ['ssh_key']}
        auth = NodeAuthPassword('password')
        self.assertRaises(LibcloudError, n._get_and_check_auth, auth)

    def test_get_auth_password(self):
        n = NodeDriver('foo')
        n.features = {'create_node': ['password']}
        auth = NodeAuthPassword('password')
        self.assertEqual(auth, n._get_and_check_auth(auth))

    def test_get_auth_password_but_given_ssh_key(self):
        n = NodeDriver('foo')
        n.features = {'create_node': ['password']}
        auth = NodeAuthSSHKey('publickey')
        self.assertRaises(LibcloudError, n._get_and_check_auth, auth)

    def test_get_auth_default_ssh_key(self):
        n = NodeDriver('foo')
        n.features = {'create_node': ['ssh_key']}
        self.assertEqual(None, n._get_and_check_auth(None))

    def test_get_auth_default_password(self):
        n = NodeDriver('foo')
        n.features = {'create_node': ['password']}
        auth = n._get_and_check_auth(None)
        self.assertTrue(isinstance(auth, NodeAuthPassword))

    def test_get_auth_default_no_feature(self):
        n = NodeDriver('foo')
        self.assertEqual(None, n._get_and_check_auth(None))

    def test_get_auth_generates_password_but_given_nonsense(self):
        n = NodeDriver('foo')
        n.features = {'create_node': ['generates_password']}
        auth = 'nonsense'
        self.assertRaises(LibcloudError, n._get_and_check_auth, auth)

    def test_get_auth_no_features_but_given_nonsense(self):
        n = NodeDriver('foo')
        auth = 'nonsense'
        self.assertRaises(LibcloudError, n._get_and_check_auth, auth)