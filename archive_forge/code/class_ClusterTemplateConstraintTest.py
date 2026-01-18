from unittest import mock
from magnumclient import exceptions as mc_exc
from heat.engine.clients.os import magnum as mc
from heat.tests import common
from heat.tests import utils
class ClusterTemplateConstraintTest(common.HeatTestCase):

    def setUp(self):
        super(ClusterTemplateConstraintTest, self).setUp()
        self.ctx = utils.dummy_context()
        self.mock_cluster_template_get = mock.Mock()
        self.ctx.clients.client_plugin('magnum').client().cluster_templates.get = self.mock_cluster_template_get
        self.constraint = mc.ClusterTemplateConstraint()

    def test_validate(self):
        self.mock_cluster_template_get.return_value = fake_cluster_template(id='my_cluster_template')
        self.assertTrue(self.constraint.validate('my_cluster_template', self.ctx))

    def test_validate_fail(self):
        self.mock_cluster_template_get.side_effect = mc_exc.NotFound()
        self.assertFalse(self.constraint.validate('bad_cluster_template', self.ctx))