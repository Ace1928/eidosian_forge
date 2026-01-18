import base64
import datetime
import json
import os
import shutil
import tempfile
import unittest
import mock
from ruamel import yaml
from six import PY3, next
from kubernetes.client import Configuration
from .config_exception import ConfigException
from .kube_config import (ENV_KUBECONFIG_PATH_SEPARATOR, ConfigNode, FileOrData,
class TestConfigNode(BaseTestCase):
    test_obj = {'key1': 'test', 'key2': ['a', 'b', 'c'], 'key3': {'inner_key': 'inner_value'}, 'with_names': [{'name': 'test_name', 'value': 'test_value'}, {'name': 'test_name2', 'value': {'key1', 'test'}}, {'name': 'test_name3', 'value': [1, 2, 3]}], 'with_names_dup': [{'name': 'test_name', 'value': 'test_value'}, {'name': 'test_name', 'value': {'key1', 'test'}}, {'name': 'test_name3', 'value': [1, 2, 3]}]}

    def setUp(self):
        super(TestConfigNode, self).setUp()
        self.node = ConfigNode('test_obj', self.test_obj)

    def test_normal_map_array_operations(self):
        self.assertEqual('test', self.node['key1'])
        self.assertEqual(5, len(self.node))
        self.assertEqual('test_obj/key2', self.node['key2'].name)
        self.assertEqual(['a', 'b', 'c'], self.node['key2'].value)
        self.assertEqual('b', self.node['key2'][1])
        self.assertEqual(3, len(self.node['key2']))
        self.assertEqual('test_obj/key3', self.node['key3'].name)
        self.assertEqual({'inner_key': 'inner_value'}, self.node['key3'].value)
        self.assertEqual('inner_value', self.node['key3']['inner_key'])
        self.assertEqual(1, len(self.node['key3']))

    def test_get_with_name(self):
        node = self.node['with_names']
        self.assertEqual('test_value', node.get_with_name('test_name')['value'])
        self.assertTrue(isinstance(node.get_with_name('test_name2'), ConfigNode))
        self.assertTrue(isinstance(node.get_with_name('test_name3'), ConfigNode))
        self.assertEqual('test_obj/with_names[name=test_name2]', node.get_with_name('test_name2').name)
        self.assertEqual('test_obj/with_names[name=test_name3]', node.get_with_name('test_name3').name)

    def test_key_does_not_exists(self):
        self.expect_exception(lambda: self.node['not-exists-key'], 'Expected key not-exists-key in test_obj')
        self.expect_exception(lambda: self.node['key3']['not-exists-key'], 'Expected key not-exists-key in test_obj/key3')

    def test_get_with_name_on_invalid_object(self):
        self.expect_exception(lambda: self.node['key2'].get_with_name('no-name'), "Expected all values in test_obj/key2 list to have 'name' key")

    def test_get_with_name_on_non_list_object(self):
        self.expect_exception(lambda: self.node['key3'].get_with_name('no-name'), 'Expected test_obj/key3 to be a list')

    def test_get_with_name_on_name_does_not_exists(self):
        self.expect_exception(lambda: self.node['with_names'].get_with_name('no-name'), 'Expected object with name no-name in test_obj/with_names list')

    def test_get_with_name_on_duplicate_name(self):
        self.expect_exception(lambda: self.node['with_names_dup'].get_with_name('test_name'), 'Expected only one object with name test_name in test_obj/with_names_dup list')