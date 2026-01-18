from unittest import mock
from oslo_vmware.objects import datacenter
from oslo_vmware.tests import base
class DatacenterTestCase(base.TestCase):
    """Test the Datacenter object."""

    def test_dc(self):
        self.assertRaises(ValueError, datacenter.Datacenter, None, 'dc-1')
        self.assertRaises(ValueError, datacenter.Datacenter, mock.Mock(), None)
        dc = datacenter.Datacenter('ref', 'name')
        self.assertEqual('ref', dc.ref)
        self.assertEqual('name', dc.name)