import copy
import testtools
from magnumclient import exceptions
from magnumclient.tests import utils
from magnumclient.v1 import certificates
class CertificateManagerTest(testtools.TestCase):

    def setUp(self):
        super(CertificateManagerTest, self).setUp()
        self.api = utils.FakeAPI(fake_responses)
        self.mgr = certificates.CertificateManager(self.api)

    def test_cert_show_by_id(self):
        cert = self.mgr.get(CERT1['cluster_uuid'])
        expect = [('GET', '/v1/certificates/%s' % CERT1['cluster_uuid'], {}, None)]
        self.assertEqual(expect, self.api.calls)
        self.assertEqual(CERT1['cluster_uuid'], cert.cluster_uuid)
        self.assertEqual(CERT1['pem'], cert.pem)

    def test_cert_create(self):
        cert = self.mgr.create(**CREATE_CERT)
        expect = [('POST', '/v1/certificates', {}, CREATE_CERT)]
        self.assertEqual(expect, self.api.calls)
        self.assertEqual(CERT2['cluster_uuid'], cert.cluster_uuid)
        self.assertEqual(CERT2['pem'], cert.pem)
        self.assertEqual(CERT2['csr'], cert.csr)

    def test_create_fail(self):
        create_cert_fail = copy.deepcopy(CREATE_CERT)
        create_cert_fail['wrong_key'] = 'wrong'
        self.assertRaisesRegex(exceptions.InvalidAttribute, 'Key must be in %s' % ','.join(certificates.CREATION_ATTRIBUTES), self.mgr.create, **create_cert_fail)
        self.assertEqual([], self.api.calls)

    def test_rotate_ca(self):
        self.mgr.rotate_ca(cluster_uuid=CERT1['cluster_uuid'])
        expect = [('PATCH', '/v1/certificates/%s' % CERT1['cluster_uuid'], {}, None)]
        self.assertEqual(expect, self.api.calls)