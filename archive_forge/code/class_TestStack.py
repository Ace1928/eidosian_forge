from unittest import mock
from openstack import exceptions
from openstack.orchestration.v1 import stack
from openstack import resource
from openstack.tests.unit import base
from openstack.tests.unit import test_resource
class TestStack(base.TestCase):

    def test_basic(self):
        sot = stack.Stack()
        self.assertEqual('stack', sot.resource_key)
        self.assertEqual('stacks', sot.resources_key)
        self.assertEqual('/stacks', sot.base_path)
        self.assertTrue(sot.allow_create)
        self.assertTrue(sot.allow_fetch)
        self.assertTrue(sot.allow_commit)
        self.assertTrue(sot.allow_delete)
        self.assertTrue(sot.allow_list)
        self.assertDictEqual({'action': 'action', 'any_tags': 'tags-any', 'limit': 'limit', 'marker': 'marker', 'name': 'name', 'not_any_tags': 'not-tags-any', 'not_tags': 'not-tags', 'owner_id': 'owner_id', 'project_id': 'tenant_id', 'status': 'status', 'tags': 'tags', 'username': 'username'}, sot._query_mapping._mapping)

    def test_make_it(self):
        sot = stack.Stack(**FAKE)
        self.assertEqual(FAKE['capabilities'], sot.capabilities)
        self.assertEqual(FAKE['creation_time'], sot.created_at)
        self.assertEqual(FAKE['deletion_time'], sot.deleted_at)
        self.assertEqual(FAKE['description'], sot.description)
        self.assertEqual(FAKE['environment'], sot.environment)
        self.assertEqual(FAKE['environment_files'], sot.environment_files)
        self.assertEqual(FAKE['files'], sot.files)
        self.assertEqual(FAKE['files_container'], sot.files_container)
        self.assertTrue(sot.is_rollback_disabled)
        self.assertEqual(FAKE['id'], sot.id)
        self.assertEqual(FAKE['links'], sot.links)
        self.assertEqual(FAKE['notification_topics'], sot.notification_topics)
        self.assertEqual(FAKE['outputs'], sot.outputs)
        self.assertEqual(FAKE['parameters'], sot.parameters)
        self.assertEqual(FAKE['name'], sot.name)
        self.assertEqual(FAKE['status'], sot.status)
        self.assertEqual(FAKE['status_reason'], sot.status_reason)
        self.assertEqual(FAKE['tags'], sot.tags)
        self.assertEqual(FAKE['template_description'], sot.template_description)
        self.assertEqual(FAKE['template_url'], sot.template_url)
        self.assertEqual(FAKE['timeout_mins'], sot.timeout_mins)
        self.assertEqual(FAKE['updated_time'], sot.updated_at)

    @mock.patch.object(resource.Resource, 'create')
    def test_create(self, mock_create):
        sess = mock.Mock()
        sot = stack.Stack()
        res = sot.create(sess)
        mock_create.assert_called_once_with(sess, prepend_key=False, base_path=None)
        self.assertEqual(mock_create.return_value, res)

    @mock.patch.object(resource.Resource, 'commit')
    def test_commit(self, mock_commit):
        sess = mock.Mock()
        sot = stack.Stack()
        res = sot.commit(sess)
        mock_commit.assert_called_once_with(sess, prepend_key=False, has_body=False, base_path=None)
        self.assertEqual(mock_commit.return_value, res)

    def test_check(self):
        sess = mock.Mock()
        sot = stack.Stack(**FAKE)
        sot._action = mock.Mock()
        sot._action.side_effect = [test_resource.FakeResponse(None, 200, None), exceptions.BadRequestException(message='oops'), exceptions.NotFoundException(message='oops')]
        body = {'check': ''}
        sot.check(sess)
        sot._action.assert_called_with(sess, body)
        self.assertRaises(exceptions.BadRequestException, sot.check, sess)
        self.assertRaises(exceptions.NotFoundException, sot.check, sess)

    def test_fetch(self):
        sess = mock.Mock()
        sess.default_microversion = None
        sot = stack.Stack(**FAKE)
        sess.get = mock.Mock()
        sess.get.side_effect = [test_resource.FakeResponse({'stack': {'stack_status': 'CREATE_COMPLETE'}}, 200), test_resource.FakeResponse({'stack': {'stack_status': 'CREATE_COMPLETE'}}, 200), exceptions.ResourceNotFound(message='oops'), test_resource.FakeResponse({'stack': {'stack_status': 'DELETE_COMPLETE'}}, 200)]
        self.assertEqual(sot, sot.fetch(sess))
        sess.get.assert_called_with('stacks/{id}'.format(id=sot.id), microversion=None, skip_cache=False)
        sot.fetch(sess, resolve_outputs=False)
        sess.get.assert_called_with('stacks/{id}?resolve_outputs=False'.format(id=sot.id), microversion=None, skip_cache=False)
        ex = self.assertRaises(exceptions.ResourceNotFound, sot.fetch, sess)
        self.assertEqual('oops', str(ex))
        ex = self.assertRaises(exceptions.ResourceNotFound, sot.fetch, sess)
        self.assertEqual('No stack found for %s' % FAKE_ID, str(ex))

    def test_abandon(self):
        sess = mock.Mock()
        sess.default_microversion = None
        mock_response = mock.Mock()
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.json.return_value = {}
        sess.delete = mock.Mock(return_value=mock_response)
        sot = stack.Stack(**FAKE)
        sot.abandon(sess)
        sess.delete.assert_called_with('stacks/%s/%s/abandon' % (FAKE_NAME, FAKE_ID))

    def test_export(self):
        sess = mock.Mock()
        sess.default_microversion = None
        mock_response = mock.Mock()
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.json.return_value = {}
        sess.get = mock.Mock(return_value=mock_response)
        sot = stack.Stack(**FAKE)
        sot.export(sess)
        sess.get.assert_called_with('stacks/%s/%s/export' % (FAKE_NAME, FAKE_ID))

    def test_update(self):
        sess = mock.Mock()
        sess.default_microversion = None
        mock_response = mock.Mock()
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.json.return_value = {}
        sess.put = mock.Mock(return_value=mock_response)
        sot = stack.Stack(**FAKE)
        body = sot._body.dirty.copy()
        sot.update(sess)
        sess.put.assert_called_with('/stacks/%s/%s' % (FAKE_NAME, FAKE_ID), headers={}, microversion=None, json=body)

    def test_update_preview(self):
        sess = mock.Mock()
        sess.default_microversion = None
        mock_response = mock.Mock()
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.json.return_value = FAKE_UPDATE_PREVIEW_RESPONSE.copy()
        sess.put = mock.Mock(return_value=mock_response)
        sot = stack.Stack(**FAKE)
        body = sot._body.dirty.copy()
        ret = sot.update(sess, preview=True)
        sess.put.assert_called_with('stacks/%s/%s/preview' % (FAKE_NAME, FAKE_ID), headers={}, microversion=None, json=body)
        self.assertEqual(FAKE_UPDATE_PREVIEW_RESPONSE['added'], ret.added)
        self.assertEqual(FAKE_UPDATE_PREVIEW_RESPONSE['deleted'], ret.deleted)
        self.assertEqual(FAKE_UPDATE_PREVIEW_RESPONSE['replaced'], ret.replaced)
        self.assertEqual(FAKE_UPDATE_PREVIEW_RESPONSE['unchanged'], ret.unchanged)
        self.assertEqual(FAKE_UPDATE_PREVIEW_RESPONSE['updated'], ret.updated)

    def test_suspend(self):
        sess = mock.Mock()
        mock_response = mock.Mock()
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.json.return_value = {}
        sess.post = mock.Mock(return_value=mock_response)
        url = 'stacks/%s/actions' % FAKE_ID
        body = {'suspend': None}
        sot = stack.Stack(**FAKE)
        res = sot.suspend(sess)
        self.assertIsNone(res)
        sess.post.assert_called_with(url, json=body, microversion=None)

    def test_resume(self):
        sess = mock.Mock()
        mock_response = mock.Mock()
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.json.return_value = {}
        sess.post = mock.Mock(return_value=mock_response)
        url = 'stacks/%s/actions' % FAKE_ID
        body = {'resume': None}
        sot = stack.Stack(**FAKE)
        res = sot.resume(sess)
        self.assertIsNone(res)
        sess.post.assert_called_with(url, json=body, microversion=None)