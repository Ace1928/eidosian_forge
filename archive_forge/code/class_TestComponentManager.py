import itertools
import json
import logging
from unittest import mock
from keystoneauth1 import adapter
import requests
from openstack import exceptions
from openstack import format
from openstack import resource
from openstack.tests.unit import base
from openstack import utils
class TestComponentManager(base.TestCase):

    def test_create_basic(self):
        sot = resource._ComponentManager()
        self.assertEqual(dict(), sot.attributes)
        self.assertEqual(set(), sot._dirty)

    def test_create_unsynced(self):
        attrs = {'hey': 1, 'hi': 2, 'hello': 3}
        sync = False
        sot = resource._ComponentManager(attributes=attrs, synchronized=sync)
        self.assertEqual(attrs, sot.attributes)
        self.assertEqual(set(attrs.keys()), sot._dirty)

    def test_create_synced(self):
        attrs = {'hey': 1, 'hi': 2, 'hello': 3}
        sync = True
        sot = resource._ComponentManager(attributes=attrs, synchronized=sync)
        self.assertEqual(attrs, sot.attributes)
        self.assertEqual(set(), sot._dirty)

    def test_getitem(self):
        key = 'key'
        value = 'value'
        attrs = {key: value}
        sot = resource._ComponentManager(attributes=attrs)
        self.assertEqual(value, sot.__getitem__(key))

    def test_setitem_new(self):
        key = 'key'
        value = 'value'
        sot = resource._ComponentManager()
        sot.__setitem__(key, value)
        self.assertIn(key, sot.attributes)
        self.assertIn(key, sot.dirty)

    def test_setitem_unchanged(self):
        key = 'key'
        value = 'value'
        attrs = {key: value}
        sot = resource._ComponentManager(attributes=attrs, synchronized=True)
        sot.__setitem__(key, value)
        self.assertEqual(value, sot.attributes[key])
        self.assertNotIn(key, sot.dirty)

    def test_delitem(self):
        key = 'key'
        value = 'value'
        attrs = {key: value}
        sot = resource._ComponentManager(attributes=attrs, synchronized=True)
        sot.__delitem__(key)
        self.assertIsNone(sot.dirty[key])

    def test_iter(self):
        attrs = {'key': 'value'}
        sot = resource._ComponentManager(attributes=attrs)
        self.assertCountEqual(iter(attrs), sot.__iter__())

    def test_len(self):
        attrs = {'key': 'value'}
        sot = resource._ComponentManager(attributes=attrs)
        self.assertEqual(len(attrs), sot.__len__())

    def test_dirty(self):
        key = 'key'
        key2 = 'key2'
        value = 'value'
        attrs = {key: value}
        sot = resource._ComponentManager(attributes=attrs, synchronized=False)
        self.assertEqual({key: value}, sot.dirty)
        sot.__setitem__(key2, value)
        self.assertEqual({key: value, key2: value}, sot.dirty)

    def test_clean(self):
        key = 'key'
        value = 'value'
        attrs = {key: value}
        sot = resource._ComponentManager(attributes=attrs, synchronized=False)
        self.assertEqual(attrs, sot.dirty)
        sot.clean()
        self.assertEqual(dict(), sot.dirty)