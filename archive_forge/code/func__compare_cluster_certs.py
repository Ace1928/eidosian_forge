from openstack.container_infrastructure_management.v1 import (
from openstack.tests.unit import base
def _compare_cluster_certs(self, exp, real):
    self.assertDictEqual(cluster_certificate.ClusterCertificate(**exp).to_dict(computed=False), real.to_dict(computed=False))