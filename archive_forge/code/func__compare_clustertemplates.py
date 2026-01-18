import testtools
from openstack.container_infrastructure_management.v1 import cluster_template
from openstack import exceptions
from openstack.tests.unit import base
def _compare_clustertemplates(self, exp, real):
    self.assertDictEqual(cluster_template.ClusterTemplate(**exp).to_dict(computed=False), real.to_dict(computed=False))