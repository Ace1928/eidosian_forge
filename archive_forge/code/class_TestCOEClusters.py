from openstack.container_infrastructure_management.v1 import (
from openstack.tests.unit import base
class TestCOEClusters(base.TestCase):

    def _compare_cluster_certs(self, exp, real):
        self.assertDictEqual(cluster_certificate.ClusterCertificate(**exp).to_dict(computed=False), real.to_dict(computed=False))

    def get_mock_url(self, service_type='container-infrastructure-management', base_url_append=None, append=None, resource=None):
        return super(TestCOEClusters, self).get_mock_url(service_type=service_type, resource=resource, append=append, base_url_append=base_url_append)

    def test_get_coe_cluster_certificate(self):
        self.register_uris([dict(method='GET', uri=self.get_mock_url(resource='certificates', append=[coe_cluster_ca_obj['cluster_uuid']]), json=coe_cluster_ca_obj)])
        ca_cert = self.cloud.get_coe_cluster_certificate(coe_cluster_ca_obj['cluster_uuid'])
        self._compare_cluster_certs(coe_cluster_ca_obj, ca_cert)
        self.assert_calls()

    def test_sign_coe_cluster_certificate(self):
        self.register_uris([dict(method='POST', uri=self.get_mock_url(resource='certificates'), json={'cluster_uuid': coe_cluster_signed_cert_obj['cluster_uuid'], 'csr': coe_cluster_signed_cert_obj['csr']})])
        self.cloud.sign_coe_cluster_certificate(coe_cluster_signed_cert_obj['cluster_uuid'], coe_cluster_signed_cert_obj['csr'])
        self.assert_calls()