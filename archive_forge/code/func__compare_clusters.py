import copy
from openstack.clustering.v1 import cluster
from openstack.tests.unit import base
def _compare_clusters(self, exp, real):
    self.assertEqual(cluster.Cluster(**exp).to_dict(computed=False), real.to_dict(computed=False))