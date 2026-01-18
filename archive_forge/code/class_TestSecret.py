from unittest import mock
from openstack.key_manager.v1 import secret
from openstack.tests.unit import base
class TestSecret(base.TestCase):

    def test_basic(self):
        sot = secret.Secret()
        self.assertIsNone(sot.resource_key)
        self.assertEqual('secrets', sot.resources_key)
        self.assertEqual('/secrets', sot.base_path)
        self.assertTrue(sot.allow_create)
        self.assertTrue(sot.allow_fetch)
        self.assertTrue(sot.allow_commit)
        self.assertTrue(sot.allow_delete)
        self.assertTrue(sot.allow_list)
        self.assertDictEqual({'name': 'name', 'mode': 'mode', 'bits': 'bits', 'secret_type': 'secret_type', 'acl_only': 'acl_only', 'created': 'created', 'updated': 'updated', 'expiration': 'expiration', 'sort': 'sort', 'algorithm': 'alg', 'limit': 'limit', 'marker': 'marker'}, sot._query_mapping._mapping)

    def test_make_it(self):
        sot = secret.Secret(**EXAMPLE)
        self.assertEqual(EXAMPLE['algorithm'], sot.algorithm)
        self.assertEqual(EXAMPLE['bit_length'], sot.bit_length)
        self.assertEqual(EXAMPLE['content_types'], sot.content_types)
        self.assertEqual(EXAMPLE['expiration'], sot.expires_at)
        self.assertEqual(EXAMPLE['mode'], sot.mode)
        self.assertEqual(EXAMPLE['name'], sot.name)
        self.assertEqual(EXAMPLE['secret_ref'], sot.secret_ref)
        self.assertEqual(EXAMPLE['secret_ref'], sot.id)
        self.assertEqual(ID_VAL, sot.secret_id)
        self.assertEqual(EXAMPLE['status'], sot.status)
        self.assertEqual(EXAMPLE['updated'], sot.updated_at)
        self.assertEqual(EXAMPLE['secret_type'], sot.secret_type)
        self.assertEqual(EXAMPLE['payload'], sot.payload)
        self.assertEqual(EXAMPLE['payload_content_type'], sot.payload_content_type)
        self.assertEqual(EXAMPLE['payload_content_encoding'], sot.payload_content_encoding)

    def test_get_no_payload(self):
        sot = secret.Secret(id='id')
        sess = mock.Mock()
        rv = mock.Mock()
        return_body = {'status': 'cool'}
        rv.json = mock.Mock(return_value=return_body)
        sess.get = mock.Mock(return_value=rv)
        sot.fetch(sess)
        sess.get.assert_called_once_with('secrets/id')

    def _test_payload(self, sot, metadata, content_type):
        content_type = 'some/type'
        metadata_response = mock.Mock()
        metadata_response.json = mock.Mock(return_value=metadata.copy())
        payload_response = mock.Mock()
        payload = 'secret info'
        payload_response.text = payload
        sess = mock.Mock()
        sess.get = mock.Mock(side_effect=[metadata_response, payload_response])
        rv = sot.fetch(sess)
        sess.get.assert_has_calls([mock.call('secrets/id'), mock.call('secrets/id/payload', headers={'Accept': content_type}, skip_cache=False)])
        self.assertEqual(rv.payload, payload)
        self.assertEqual(rv.status, metadata['status'])

    def test_get_with_payload_from_argument(self):
        metadata = {'status': 'great'}
        content_type = 'some/type'
        sot = secret.Secret(id='id', payload_content_type=content_type)
        self._test_payload(sot, metadata, content_type)

    def test_get_with_payload_from_content_types(self):
        content_type = 'some/type'
        metadata = {'status': 'fine', 'content_types': {'default': content_type}}
        sot = secret.Secret(id='id')
        self._test_payload(sot, metadata, content_type)