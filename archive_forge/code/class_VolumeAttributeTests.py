from tests.compat import mock, unittest
from boto.ec2.snapshot import Snapshot
from boto.ec2.tag import Tag, TagSet
from boto.ec2.volume import Volume, AttachmentSet, VolumeAttribute
class VolumeAttributeTests(unittest.TestCase):

    def setUp(self):
        self.volume_attribute = VolumeAttribute()
        self.volume_attribute._key_name = 'key_name'
        self.volume_attribute.attrs = {'key_name': False}

    def test_startElement_with_name_autoEnableIO_sets_key_name(self):
        self.volume_attribute.startElement('autoEnableIO', None, None)
        self.assertEqual(self.volume_attribute._key_name, 'autoEnableIO')

    def test_startElement_without_name_autoEnableIO_returns_None(self):
        retval = self.volume_attribute.startElement('some name', None, None)
        self.assertEqual(retval, None)

    def test_endElement_with_name_value_and_value_true_sets_attrs_key_name_True(self):
        self.volume_attribute.endElement('value', 'true', None)
        self.assertEqual(self.volume_attribute.attrs['key_name'], True)

    def test_endElement_with_name_value_and_value_false_sets_attrs_key_name_False(self):
        self.volume_attribute._key_name = 'other_key_name'
        self.volume_attribute.endElement('value', 'false', None)
        self.assertEqual(self.volume_attribute.attrs['other_key_name'], False)

    def test_endElement_with_name_volumeId_sets_id(self):
        self.volume_attribute.endElement('volumeId', 'some_value', None)
        self.assertEqual(self.volume_attribute.id, 'some_value')

    def test_endElement_with_other_name_sets_other_name_attribute(self):
        self.volume_attribute.endElement('someName', 'some value', None)
        self.assertEqual(self.volume_attribute.someName, 'some value')