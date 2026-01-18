from unittest import mock
from heat.api.openstack.v1.views import stacks_view
from heat.common import identifier
from heat.tests import common
class TestStacksViewBuilder(common.HeatTestCase):

    def setUp(self):
        super(TestStacksViewBuilder, self).setUp()
        self.request = mock.Mock()
        self.request.params = {}
        identity = identifier.HeatIdentifier('123456', 'wordpress', '1')
        self.stack1 = {u'stack_identity': dict(identity), u'updated_time': u'2012-07-09T09:13:11Z', u'template_description': u'blah', u'description': u'blah', u'stack_status_reason': u'Stack successfully created', u'creation_time': u'2012-07-09T09:12:45Z', u'stack_name': identity.stack_name, u'stack_action': u'CREATE', u'stack_status': u'COMPLETE', u'parameters': {'foo': 'bar'}, u'outputs': ['key', 'value'], u'notification_topics': [], u'capabilities': [], u'disable_rollback': True, u'timeout_mins': 60}

    def test_stack_index(self):
        stacks = [self.stack1]
        stack_view = stacks_view.collection(self.request, stacks)
        self.assertIn('stacks', stack_view)
        self.assertEqual(1, len(stack_view['stacks']))

    @mock.patch.object(stacks_view, 'format_stack')
    def test_stack_basic_details(self, mock_format_stack):
        stacks = [self.stack1]
        expected_keys = stacks_view.basic_keys
        stacks_view.collection(self.request, stacks)
        mock_format_stack.assert_called_once_with(self.request, self.stack1, expected_keys, mock.ANY)

    @mock.patch.object(stacks_view.views_common, 'get_collection_links')
    def test_append_collection_links(self, mock_get_collection_links):
        stacks = [self.stack1]
        mock_get_collection_links.return_value = 'fake links'
        stack_view = stacks_view.collection(self.request, stacks)
        self.assertIn('links', stack_view)

    @mock.patch.object(stacks_view.views_common, 'get_collection_links')
    def test_doesnt_append_collection_links(self, mock_get_collection_links):
        stacks = [self.stack1]
        mock_get_collection_links.return_value = None
        stack_view = stacks_view.collection(self.request, stacks)
        self.assertNotIn('links', stack_view)

    @mock.patch.object(stacks_view.views_common, 'get_collection_links')
    def test_append_collection_count(self, mock_get_collection_links):
        stacks = [self.stack1]
        count = 1
        stack_view = stacks_view.collection(self.request, stacks, count)
        self.assertIn('count', stack_view)
        self.assertEqual(1, stack_view['count'])

    @mock.patch.object(stacks_view.views_common, 'get_collection_links')
    def test_doesnt_append_collection_count(self, mock_get_collection_links):
        stacks = [self.stack1]
        stack_view = stacks_view.collection(self.request, stacks)
        self.assertNotIn('count', stack_view)

    @mock.patch.object(stacks_view.views_common, 'get_collection_links')
    def test_appends_collection_count_of_zero(self, mock_get_collection_links):
        stacks = [self.stack1]
        count = 0
        stack_view = stacks_view.collection(self.request, stacks, count)
        self.assertIn('count', stack_view)
        self.assertEqual(0, stack_view['count'])