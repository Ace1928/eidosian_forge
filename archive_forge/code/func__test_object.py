import os
import tempfile
import uuid
from openstackclient.tests.functional.object.v1 import common
def _test_object(self, object_file):
    raw_output = self.openstack('container create ' + self.CONTAINER_NAME)
    items = self.parse_listing(raw_output)
    self.assert_show_fields(items, CONTAINER_FIELDS)
    raw_output = self.openstack('container list')
    items = self.parse_listing(raw_output)
    self.assert_table_structure(items, BASIC_LIST_HEADERS)
    self.openstack('container show ' + self.CONTAINER_NAME)
    self.openstack('container save ' + self.CONTAINER_NAME)
    raw_output = self.openstack('object create %s %s' % (self.CONTAINER_NAME, object_file))
    items = self.parse_listing(raw_output)
    self.assert_show_fields(items, OBJECT_FIELDS)
    raw_output = self.openstack('object list %s' % self.CONTAINER_NAME)
    items = self.parse_listing(raw_output)
    self.assert_table_structure(items, BASIC_LIST_HEADERS)
    self.openstack('object save %s %s' % (self.CONTAINER_NAME, object_file))
    tmp_file = 'tmp.txt'
    self.addCleanup(os.remove, tmp_file)
    self.openstack('object save %s %s --file %s' % (self.CONTAINER_NAME, object_file, tmp_file))
    raw_output = self.openstack('object save %s %s --file -' % (self.CONTAINER_NAME, object_file))
    self.assertEqual(raw_output, 'test content')
    self.openstack('object show %s %s' % (self.CONTAINER_NAME, object_file))
    raw_output = self.openstack('object delete %s %s' % (self.CONTAINER_NAME, object_file))
    self.assertEqual(0, len(raw_output))
    self.openstack('object create %s %s' % (self.CONTAINER_NAME, object_file))
    raw_output = self.openstack('container delete -r %s' % self.CONTAINER_NAME)
    self.assertEqual(0, len(raw_output))