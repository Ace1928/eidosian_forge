from unittest import mock
from heat.common import exception
from heat.engine.resources.openstack.designate import zone
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def _test_resource_validate(self, type_, prp):

    def _side_effect(key):
        if key == prp:
            return None
        if key == zone.DesignateZone.TYPE:
            return type_
        else:
            return sample_template['resources']['test_resource']['properties'][key]
    self.test_resource.properties = mock.MagicMock()
    self.test_resource.properties.get.side_effect = _side_effect
    self.test_resource.properties.__getitem__.side_effect = _side_effect
    ex = self.assertRaises(exception.StackValidationFailed, self.test_resource.validate)
    self.assertEqual('Property %s is required for zone type %s' % (prp, type_), ex.message)