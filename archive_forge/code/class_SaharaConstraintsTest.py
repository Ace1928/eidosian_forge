from unittest import mock
import uuid
from saharaclient.api import base as sahara_base
from heat.common import exception
from heat.engine.clients.os import sahara
from heat.tests import common
from heat.tests import utils
class SaharaConstraintsTest(common.HeatTestCase):
    scenarios = [('JobType', dict(constraint=sahara.JobTypeConstraint(), resource_name=None)), ('ClusterTemplate', dict(constraint=sahara.ClusterTemplateConstraint(), resource_name='cluster_templates')), ('DataSource', dict(constraint=sahara.DataSourceConstraint(), resource_name='data_sources')), ('Cluster', dict(constraint=sahara.ClusterConstraint(), resource_name='clusters')), ('JobBinary', dict(constraint=sahara.JobBinaryConstraint(), resource_name='job_binaries')), ('Plugin', dict(constraint=sahara.PluginConstraint(), resource_name=None)), ('Image', dict(constraint=sahara.ImageConstraint(), resource_name='images'))]

    def setUp(self):
        super(SaharaConstraintsTest, self).setUp()
        self.ctx = utils.dummy_context()
        self.mock_get = mock.Mock()
        cl_plgn = self.ctx.clients.client_plugin('sahara')
        cl_plgn.find_resource_by_name_or_id = self.mock_get
        cl_plgn.get_image_id = self.mock_get
        cl_plgn.get_plugin_id = self.mock_get
        cl_plgn.get_job_type = self.mock_get

    def test_validation(self):
        self.mock_get.return_value = 'fake_val'
        self.assertTrue(self.constraint.validate('foo', self.ctx))
        if self.resource_name is None:
            self.mock_get.assert_called_once_with('foo')
        else:
            self.mock_get.assert_called_once_with(self.resource_name, 'foo')

    def test_validation_error(self):
        self.mock_get.side_effect = exception.EntityNotFound(entity='Fake entity', name='bar')
        self.assertFalse(self.constraint.validate('bar', self.ctx))
        if self.resource_name is None:
            self.mock_get.assert_called_once_with('bar')
        else:
            self.mock_get.assert_called_once_with(self.resource_name, 'bar')