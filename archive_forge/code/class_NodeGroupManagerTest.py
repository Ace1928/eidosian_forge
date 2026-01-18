import copy
import testtools
from testtools import matchers
from magnumclient import exceptions
from magnumclient.tests import utils
from magnumclient.v1 import nodegroups
class NodeGroupManagerTest(testtools.TestCase):

    def setUp(self):
        super(NodeGroupManagerTest, self).setUp()
        self.api = utils.FakeAPI(fake_responses)
        self.mgr = nodegroups.NodeGroupManager(self.api)
        self.cluster_id = 'test'
        self.base_path = '/v1/clusters/test/nodegroups/'

    def test_nodegroup_list(self):
        clusters = self.mgr.list(self.cluster_id)
        expect = [('GET', self.base_path, {}, None)]
        self.assertEqual(expect, self.api.calls)
        self.assertThat(clusters, matchers.HasLength(2))

    def _test_nodegroup_list_with_filters(self, cluster_id, limit=None, marker=None, sort_key=None, sort_dir=None, detail=False, expect=[]):
        nodegroup_filter = self.mgr.list(cluster_id, limit=limit, marker=marker, sort_key=sort_key, sort_dir=sort_dir, detail=detail)
        self.assertEqual(expect, self.api.calls)
        self.assertThat(nodegroup_filter, matchers.HasLength(2))

    def test_nodegroup_list_with_limit(self):
        expect = [('GET', self.base_path + '?limit=2', {}, None)]
        self._test_nodegroup_list_with_filters(self.cluster_id, limit=2, expect=expect)

    def test_nodegroup_list_with_marker(self):
        filter_ = '?marker=%s' % NODEGROUP2['uuid']
        expect = [('GET', self.base_path + filter_, {}, None)]
        self._test_nodegroup_list_with_filters(self.cluster_id, marker=NODEGROUP2['uuid'], expect=expect)

    def test_nodegroup_list_with_marker_limit(self):
        filter_ = '?limit=2&marker=%s' % NODEGROUP2['uuid']
        expect = [('GET', self.base_path + filter_, {}, None)]
        self._test_nodegroup_list_with_filters(self.cluster_id, limit=2, marker=NODEGROUP2['uuid'], expect=expect)

    def test_nodegroup_list_with_sort_dir(self):
        expect = [('GET', '/v1/clusters/test/nodegroups/?sort_dir=asc', {}, None)]
        self._test_nodegroup_list_with_filters(self.cluster_id, sort_dir='asc', expect=expect)

    def test_nodegroup_list_with_sort_key(self):
        expect = [('GET', '/v1/clusters/test/nodegroups/?sort_key=uuid', {}, None)]
        self._test_nodegroup_list_with_filters(self.cluster_id, sort_key='uuid', expect=expect)

    def test_nodegroup_list_with_sort_key_dir(self):
        expect = [('GET', self.base_path + '?sort_key=uuid&sort_dir=desc', {}, None)]
        self._test_nodegroup_list_with_filters(self.cluster_id, sort_key='uuid', sort_dir='desc', expect=expect)

    def test_nodegroup_show_by_name(self):
        nodegroup = self.mgr.get(self.cluster_id, NODEGROUP1['name'])
        expect = [('GET', self.base_path + '%s' % NODEGROUP1['name'], {}, None)]
        self.assertEqual(expect, self.api.calls)
        self.assertEqual(NODEGROUP1['name'], nodegroup.name)

    def test_nodegroup_show_by_id(self):
        nodegroup = self.mgr.get(self.cluster_id, NODEGROUP1['id'])
        expect = [('GET', self.base_path + '%s' % NODEGROUP1['id'], {}, None)]
        self.assertEqual(expect, self.api.calls)
        self.assertEqual(NODEGROUP1['name'], nodegroup.name)

    def test_nodegroup_delete_by_id(self):
        nodegroup = self.mgr.delete(self.cluster_id, NODEGROUP1['id'])
        expect = [('DELETE', self.base_path + '%s' % NODEGROUP1['id'], {}, None)]
        self.assertEqual(expect, self.api.calls)
        self.assertIsNone(nodegroup)

    def test_nodegroup_delete_by_name(self):
        nodegroup = self.mgr.delete(self.cluster_id, NODEGROUP1['name'])
        expect = [('DELETE', self.base_path + '%s' % NODEGROUP1['name'], {}, None)]
        self.assertEqual(expect, self.api.calls)
        self.assertIsNone(nodegroup)

    def test_nodegroup_update(self):
        patch = {'op': 'replace', 'value': NEW_NODE_COUNT, 'path': '/node_count'}
        nodegroup = self.mgr.update(self.cluster_id, id=NODEGROUP1['id'], patch=patch)
        expect = [('PATCH', self.base_path + '%s' % NODEGROUP1['id'], {}, patch)]
        self.assertEqual(expect, self.api.calls)
        self.assertEqual(NEW_NODE_COUNT, nodegroup.node_count)

    def test_nodegroup_create(self):
        nodegroup = self.mgr.create(self.cluster_id, **CREATE_NODEGROUP)
        expect = [('POST', self.base_path, {}, CREATE_NODEGROUP)]
        self.assertEqual(expect, self.api.calls)
        self.assertTrue(nodegroup)

    def test_nodegroup_create_with_docker_volume_size(self):
        ng_with_volume_size = dict()
        ng_with_volume_size.update(CREATE_NODEGROUP)
        ng_with_volume_size['docker_volume_size'] = 20
        nodegroup = self.mgr.create(self.cluster_id, **ng_with_volume_size)
        expect = [('POST', self.base_path, {}, ng_with_volume_size)]
        self.assertEqual(expect, self.api.calls)
        self.assertTrue(nodegroup)

    def test_nodegroup_create_with_labels(self):
        ng_with_labels = dict()
        ng_with_labels.update(CREATE_NODEGROUP)
        ng_with_labels['labels'] = 'key=val'
        nodegroup = self.mgr.create(self.cluster_id, **ng_with_labels)
        expect = [('POST', self.base_path, {}, ng_with_labels)]
        self.assertEqual(expect, self.api.calls)
        self.assertTrue(nodegroup)

    def test_nodegroup_create_fail(self):
        CREATE_NODEGROUP_FAIL = copy.deepcopy(CREATE_NODEGROUP)
        CREATE_NODEGROUP_FAIL['wrong_key'] = 'wrong'
        self.assertRaisesRegex(exceptions.InvalidAttribute, 'Key must be in %s' % ','.join(nodegroups.CREATION_ATTRIBUTES), self.mgr.create, self.cluster_id, **CREATE_NODEGROUP_FAIL)
        self.assertEqual([], self.api.calls)