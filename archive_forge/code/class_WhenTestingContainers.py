from unittest import mock
from oslo_serialization import jsonutils
from oslo_utils import timeutils
from barbicanclient import base
from barbicanclient.tests import test_client
from barbicanclient.v1 import acls
from barbicanclient.v1 import containers
from barbicanclient.v1 import secrets
class WhenTestingContainers(test_client.BaseEntityResource):

    def setUp(self):
        self._setUp('containers')
        self.container = ContainerData()
        self.manager = self.client.containers
        self.consumers_post_resource = self.entity_href + '/consumers/'
        self.consumers_delete_resource = self.entity_href + '/consumers'

    def test_should_generic_container_str(self):
        container_obj = self.manager.create(name=self.container.name)
        self.assertIn(self.container.name, str(container_obj))
        self.assertIn(' generic ', str(container_obj))

    def test_should_certificate_container_str(self):
        container_obj = self.manager.create_certificate(name=self.container.name)
        self.assertIn(self.container.name, str(container_obj))
        self.assertIn(' certificate ', str(container_obj))

    def test_should_rsa_container_str(self):
        container_obj = self.manager.create_rsa(name=self.container.name)
        self.assertIn(self.container.name, str(container_obj))
        self.assertIn(' rsa ', str(container_obj))

    def test_should_generic_container_repr(self):
        container_obj = self.manager.create(name=self.container.name)
        self.assertIn('name="{0}"'.format(self.container.name), repr(container_obj))

    def test_should_certificate_container_repr(self):
        container_obj = self.manager.create_certificate(name=self.container.name)
        self.assertIn('name="{0}"'.format(self.container.name), repr(container_obj))

    def test_should_rsa_container_repr(self):
        container_obj = self.manager.create_rsa(name=self.container.name)
        self.assertIn('name="{0}"'.format(self.container.name), repr(container_obj))

    def test_should_store_generic_via_constructor(self):
        data = {'container_ref': self.entity_href}
        self.responses.post(self.entity_base + '/', json=data)
        container = self.manager.create(name=self.container.name, secrets=self.container.generic_secrets)
        container_href = container.store()
        self.assertEqual(self.entity_href, container_href)
        self.assertEqual(self.entity_base + '/', self.responses.last_request.url)
        container_req = jsonutils.loads(self.responses.last_request.text)
        self.assertEqual(self.container.name, container_req['name'])
        self.assertEqual(self.container.type, container_req['type'])
        self.assertEqual(self.container.generic_secret_refs_json, container_req['secret_refs'])

    def test_should_store_generic_via_attributes(self):
        data = {'container_ref': self.entity_href}
        self.responses.post(self.entity_base + '/', json=data)
        container = self.manager.create()
        container.name = self.container.name
        container.add(self.container.secret.name, self.container.secret)
        container_href = container.store()
        self.assertEqual(self.entity_href, container_href)
        self.assertEqual(self.entity_base + '/', self.responses.last_request.url)
        container_req = jsonutils.loads(self.responses.last_request.text)
        self.assertEqual(self.container.name, container_req['name'])
        self.assertEqual(self.container.type, container_req['type'])
        self.assertEqual(self.container.generic_secret_refs_json, container_req['secret_refs'])

    def test_should_store_certificate_via_attributes(self):
        data = {'container_ref': self.entity_href}
        self.responses.post(self.entity_base + '/', json=data)
        container = self.manager.create_certificate()
        container.name = self.container.name
        container.certificate = self.container.secret
        container.private_key = self.container.secret
        container.private_key_passphrase = self.container.secret
        container.intermediates = self.container.secret
        container_href = container.store()
        self.assertEqual(self.entity_href, container_href)
        self.assertEqual(self.entity_base + '/', self.responses.last_request.url)
        container_req = jsonutils.loads(self.responses.last_request.text)
        self.assertEqual(self.container.name, container_req['name'])
        self.assertEqual('certificate', container_req['type'])
        self.assertCountEqual(self.container.certificate_secret_refs_json, container_req['secret_refs'])

    def test_should_store_certificate_via_constructor(self):
        data = {'container_ref': self.entity_href}
        self.responses.post(self.entity_base + '/', json=data)
        container = self.manager.create_certificate(name=self.container.name, certificate=self.container.secret, private_key=self.container.secret, private_key_passphrase=self.container.secret, intermediates=self.container.secret)
        container_href = container.store()
        self.assertEqual(self.entity_href, container_href)
        self.assertEqual(self.entity_base + '/', self.responses.last_request.url)
        container_req = jsonutils.loads(self.responses.last_request.text)
        self.assertEqual(self.container.name, container_req['name'])
        self.assertEqual('certificate', container_req['type'])
        self.assertCountEqual(self.container.certificate_secret_refs_json, container_req['secret_refs'])

    def test_should_store_rsa_via_attributes(self):
        data = {'container_ref': self.entity_href}
        self.responses.post(self.entity_base + '/', json=data)
        container = self.manager.create_rsa()
        container.name = self.container.name
        container.private_key = self.container.secret
        container.private_key_passphrase = self.container.secret
        container.public_key = self.container.secret
        container_href = container.store()
        self.assertEqual(self.entity_href, container_href)
        self.assertEqual(self.entity_base + '/', self.responses.last_request.url)
        container_req = jsonutils.loads(self.responses.last_request.text)
        self.assertEqual(self.container.name, container_req['name'])
        self.assertEqual('rsa', container_req['type'])
        self.assertCountEqual(self.container.rsa_secret_refs_json, container_req['secret_refs'])

    def test_should_store_rsa_via_constructor(self):
        data = {'container_ref': self.entity_href}
        self.responses.post(self.entity_base + '/', json=data)
        container = self.manager.create_rsa(name=self.container.name, private_key=self.container.secret, private_key_passphrase=self.container.secret, public_key=self.container.secret)
        container_href = container.store()
        self.assertEqual(self.entity_href, container_href)
        self.assertEqual(self.entity_base + '/', self.responses.last_request.url)
        container_req = jsonutils.loads(self.responses.last_request.text)
        self.assertEqual(self.container.name, container_req['name'])
        self.assertEqual('rsa', container_req['type'])
        self.assertCountEqual(self.container.rsa_secret_refs_json, container_req['secret_refs'])

    def test_should_get_secret_refs_when_created_using_secret_objects(self):
        data = {'container_ref': self.entity_href}
        self.responses.post(self.entity_base + '/', json=data)
        container = self.manager.create(name=self.container.name, secrets=self.container.generic_secrets)
        self.assertEqual(self.container.generic_secret_refs, container.secret_refs)

    def test_should_reload_attributes_after_store(self):
        data = {'container_ref': self.entity_href}
        self.responses.post(self.entity_base + '/', json=data)
        data = self.container.get_dict(self.entity_href)
        self.responses.get(self.entity_href, json=data)
        container = self.manager.create(name=self.container.name, secrets=self.container.generic_secrets)
        self.assertIsNone(container.status)
        self.assertIsNone(container.created)
        self.assertIsNone(container.updated)
        container_href = container.store()
        self.assertEqual(self.entity_href, container_href)
        self.assertIsNotNone(container.status)
        self.assertIsNotNone(container.created)

    def test_should_fail_add_invalid_secret_object(self):
        container = self.manager.create()
        self.assertRaises(ValueError, container.add, 'Not-a-secret', 'Actually a string')

    def test_should_fail_add_duplicate_named_secret_object(self):
        container = self.manager.create()
        container.add(self.container.secret.name, self.container.secret)
        self.assertRaises(KeyError, container.add, self.container.secret.name, self.container.secret)

    def test_should_add_remove_add_secret_object(self):
        container = self.manager.create()
        container.add(self.container.secret.name, self.container.secret)
        container.remove(self.container.secret.name)
        container.add(self.container.secret.name, self.container.secret)

    def test_should_be_immutable_after_store(self):
        data = {'container_ref': self.entity_href}
        self.responses.post(self.entity_base + '/', json=data)
        container = self.manager.create(name=self.container.name, secrets=self.container.generic_secrets)
        container_href = container.store()
        self.assertEqual(self.entity_href, container_href)
        attributes = ['name']
        for attr in attributes:
            try:
                setattr(container, attr, 'test')
                self.fail("didn't raise an ImmutableException exception")
            except base.ImmutableException:
                pass
        self.assertRaises(base.ImmutableException, container.add, self.container.secret.name, self.container.secret)

    def test_should_not_be_able_to_set_generated_attributes(self):
        container = self.manager.create()
        attributes = ['container_ref', 'created', 'updated', 'status', 'consumers']
        for attr in attributes:
            try:
                setattr(container, attr, 'test')
                self.fail("didn't raise an AttributeError exception")
            except AttributeError:
                pass

    def test_should_get_generic_container(self, container_ref=None):
        container_ref = container_ref or self.entity_href
        data = self.container.get_dict(container_ref)
        self.responses.get(self.entity_href, json=data)
        container = self.manager.get(container_ref=container_ref)
        self.assertIsInstance(container, containers.Container)
        self.assertEqual(container_ref, container.container_ref)
        self.assertEqual(self.entity_href, self.responses.last_request.url)
        self.assertIsNotNone(container.secrets)

    def test_should_get_certificate_container(self):
        data = self.container.get_dict(self.entity_href, type='certificate')
        self.responses.get(self.entity_href, json=data)
        container = self.manager.get(container_ref=self.entity_href)
        self.assertIsInstance(container, containers.Container)
        self.assertEqual(self.entity_href, container.container_ref)
        self.assertEqual(self.entity_href, self.responses.last_request.url)
        self.assertIsInstance(container, containers.CertificateContainer)
        self.assertIsNotNone(container.certificate)
        self.assertIsNotNone(container.private_key)
        self.assertIsNotNone(container.private_key_passphrase)
        self.assertIsNotNone(container.intermediates)

    def test_should_get_rsa_container(self):
        data = self.container.get_dict(self.entity_href, type='rsa')
        self.responses.get(self.entity_href, json=data)
        container = self.manager.get(container_ref=self.entity_href)
        self.assertIsInstance(container, containers.Container)
        self.assertEqual(self.entity_href, container.container_ref)
        self.assertEqual(self.entity_href, self.responses.last_request.url)
        self.assertIsInstance(container, containers.RSAContainer)
        self.assertIsNotNone(container.private_key)
        self.assertIsNotNone(container.public_key)
        self.assertIsNotNone(container.private_key_passphrase)

    def test_should_get_generic_container_using_stripped_uuid(self):
        bad_href = 'http://badsite.com/' + self.entity_id
        self.test_should_get_generic_container(bad_href)

    def test_should_get_generic_container_using_only_uuid(self):
        self.test_should_get_generic_container(self.entity_id)

    def test_should_delete_from_manager(self, container_ref=None):
        container_ref = container_ref or self.entity_href
        self.responses.delete(self.entity_href, status_code=204)
        self.manager.delete(container_ref=container_ref)
        self.assertEqual(self.entity_href, self.responses.last_request.url)

    def test_should_delete_from_manager_using_stripped_uuid(self):
        bad_href = 'http://badsite.com/' + self.entity_id
        self.test_should_delete_from_manager(bad_href)

    def test_should_delete_from_manager_using_only_uuid(self):
        self.test_should_delete_from_manager(self.entity_id)

    def test_should_delete_from_object(self, container_ref=None):
        container_ref = container_ref or self.entity_href
        data = self.container.get_dict(container_ref)
        m = self.responses.get(self.entity_href, json=data)
        n = self.responses.delete(self.entity_href, status_code=204)
        container = self.manager.get(container_ref=container_ref)
        self.assertEqual(container_ref, container.container_ref)
        container.delete()
        self.assertTrue(m.called)
        self.assertTrue(n.called)
        self.assertIsNone(container.container_ref)

    def test_should_delete_from_object_using_stripped_uuid(self):
        bad_href = 'http://badsite.com/' + self.entity_id
        self.test_should_delete_from_object(bad_href)

    def test_should_delete_from_object_using_only_uuid(self):
        self.test_should_delete_from_object(self.entity_id)

    def test_should_store_after_delete_from_object(self):
        data = self.container.get_dict(self.entity_href)
        self.responses.get(self.entity_href, json=data)
        data = self.container.get_dict(self.entity_href)
        self.responses.post(self.entity_base + '/', json=data)
        m = self.responses.delete(self.entity_href, status_code=204)
        container = self.manager.get(container_ref=self.entity_href)
        self.assertIsNotNone(container.container_ref)
        container.delete()
        self.assertEqual(self.entity_href, m.last_request.url)
        self.assertIsNone(container.container_ref)
        container.store()
        self.assertIsNotNone(container.container_ref)

    def test_should_get_list(self):
        container_resp = self.container.get_dict(self.entity_href)
        data = {'containers': [container_resp for v in range(3)]}
        self.responses.get(self.entity_base, json=data)
        containers_list = self.manager.list(limit=10, offset=5)
        self.assertTrue(len(containers_list) == 3)
        self.assertIsInstance(containers_list[0], containers.Container)
        self.assertEqual(self.entity_href, containers_list[0].container_ref)
        self.assertEqual(self.entity_base, self.responses.last_request.url.split('?')[0])
        self.assertEqual(['10'], self.responses.last_request.qs['limit'])
        self.assertEqual(['5'], self.responses.last_request.qs['offset'])

    def test_should_get_list_when_secret_ref_without_name(self):
        container_resp = self.container.get_dict(self.entity_href)
        del container_resp.get('secret_refs')[0]['name']
        data = {'containers': [container_resp for v in range(3)]}
        self.responses.get(self.entity_base, json=data)
        containers_list = self.manager.list(limit=10, offset=5)
        self.assertTrue(len(containers_list) == 3)
        self.assertIsInstance(containers_list[0], containers.Container)
        self.assertEqual(self.entity_href, containers_list[0].container_ref)
        self.assertEqual(self.entity_base, self.responses.last_request.url.split('?')[0])
        for container in containers_list:
            for name in container._secret_refs.keys():
                self.assertIsNone(name)

    def test_should_fail_get_invalid_container(self):
        self.assertRaises(ValueError, self.manager.get, **{'container_ref': '12345'})

    def test_should_fail_delete_no_href(self):
        self.assertRaises(ValueError, self.manager.delete, None)

    def test_should_register_consumer(self):
        data = self.container.get_dict(self.entity_href, consumers=[self.container.consumer])
        self.responses.post(self.entity_href + '/consumers/', json=data)
        container = self.manager.register_consumer(self.entity_href, self.container.consumer.get('name'), self.container.consumer.get('URL'))
        self.assertIsInstance(container, containers.Container)
        self.assertEqual(self.entity_href, container.container_ref)
        body = jsonutils.loads(self.responses.last_request.text)
        self.assertEqual(self.consumers_post_resource, self.responses.last_request.url)
        self.assertEqual(self.container.consumer, body)
        self.assertEqual([self.container.consumer], container.consumers)

    def test_should_remove_consumer(self):
        self.responses.delete(self.entity_href + '/consumers', status_code=204)
        self.manager.remove_consumer(self.entity_href, self.container.consumer.get('name'), self.container.consumer.get('URL'))
        body = jsonutils.loads(self.responses.last_request.text)
        self.assertEqual(self.consumers_delete_resource, self.responses.last_request.url)
        self.assertEqual(self.container.consumer, body)

    def test_should_get_total(self):
        self.responses.get(self.entity_base, json={'total': 1})
        total = self.manager.total()
        self.assertEqual(1, total)

    def test_should_get_acls_lazy(self):
        data = self.container.get_dict(self.entity_href, consumers=[self.container.consumer])
        m = self.responses.get(self.entity_href, json=data)
        acl_data = {'read': {'project-access': True, 'users': ['u2']}}
        acl_ref = self.entity_href + '/acl'
        n = self.responses.get(acl_ref, json=acl_data)
        container = self.manager.get(container_ref=self.entity_href)
        self.assertIsNotNone(container)
        self.assertEqual(self.container.name, container.name)
        self.assertTrue(m.called)
        self.assertFalse(n.called)
        self.assertEqual(['u2'], container.acls.read.users)
        self.assertTrue(container.acls.read.project_access)
        self.assertIsInstance(container.acls, acls.ContainerACL)
        self.assertEqual(acl_ref, n.last_request.url)

    def test_get_formatted_data(self):
        data = self.container.get_dict(self.entity_href)
        self.responses.get(self.entity_href, json=data)
        container = self.manager.get(container_ref=self.entity_href)
        data = container._get_formatted_data()
        self.assertEqual(self.container.name, data[1])
        self.assertEqual(timeutils.parse_isotime(self.container.created).isoformat(), data[2])