import sys
import json
from unittest.mock import Mock, call
from libcloud.test import unittest
from libcloud.compute.base import NodeSize, NodeImage, NodeLocation, NodeAuthSSHKey
from libcloud.common.upcloud import (
class TestStorageDevice(unittest.TestCase):

    def setUp(self):
        self.image = NodeImage(id='01000000-0000-4000-8000-000030060200', name='Ubuntu Server 16.04 LTS (Xenial Xerus)', driver='', extra={'type': 'template'})
        self.size = NodeSize(id='1xCPU-1GB', name='1xCPU-1GB', ram=1024, disk=30, bandwidth=2048, extra={'core_number': 1}, price=None, driver='')

    def test_storage_tier_default_value(self):
        storagedevice = _StorageDevice(self.image, self.size)
        d = storagedevice.to_dict()
        self.assertEqual(d['storage_device'][0]['tier'], 'maxiops')

    def test_storage_tier_given(self):
        self.size.extra['storage_tier'] = 'hdd'
        storagedevice = _StorageDevice(self.image, self.size)
        d = storagedevice.to_dict()
        self.assertEqual(d['storage_device'][0]['tier'], 'hdd')