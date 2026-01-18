import copy
import testtools
from testtools import matchers
from magnumclient import exceptions
from magnumclient.tests import utils
from magnumclient.v1 import clusters
class ClusterManagerTest(testtools.TestCase):

    def setUp(self):
        super(ClusterManagerTest, self).setUp()
        self.api = utils.FakeAPI(fake_responses)
        self.mgr = clusters.ClusterManager(self.api)

    def test_cluster_list(self):
        clusters = self.mgr.list()
        expect = [('GET', '/v1/clusters', {}, None)]
        self.assertEqual(expect, self.api.calls)
        self.assertThat(clusters, matchers.HasLength(2))

    def _test_cluster_list_with_filters(self, limit=None, marker=None, sort_key=None, sort_dir=None, detail=False, expect=[]):
        clusters_filter = self.mgr.list(limit=limit, marker=marker, sort_key=sort_key, sort_dir=sort_dir, detail=detail)
        self.assertEqual(expect, self.api.calls)
        self.assertThat(clusters_filter, matchers.HasLength(2))

    def test_cluster_list_with_limit(self):
        expect = [('GET', '/v1/clusters/?limit=2', {}, None)]
        self._test_cluster_list_with_filters(limit=2, expect=expect)

    def test_cluster_list_with_marker(self):
        expect = [('GET', '/v1/clusters/?marker=%s' % CLUSTER2['uuid'], {}, None)]
        self._test_cluster_list_with_filters(marker=CLUSTER2['uuid'], expect=expect)

    def test_cluster_list_with_marker_limit(self):
        expect = [('GET', '/v1/clusters/?limit=2&marker=%s' % CLUSTER2['uuid'], {}, None)]
        self._test_cluster_list_with_filters(limit=2, marker=CLUSTER2['uuid'], expect=expect)

    def test_cluster_list_with_sort_dir(self):
        expect = [('GET', '/v1/clusters/?sort_dir=asc', {}, None)]
        self._test_cluster_list_with_filters(sort_dir='asc', expect=expect)

    def test_cluster_list_with_sort_key(self):
        expect = [('GET', '/v1/clusters/?sort_key=uuid', {}, None)]
        self._test_cluster_list_with_filters(sort_key='uuid', expect=expect)

    def test_cluster_list_with_sort_key_dir(self):
        expect = [('GET', '/v1/clusters/?sort_key=uuid&sort_dir=desc', {}, None)]
        self._test_cluster_list_with_filters(sort_key='uuid', sort_dir='desc', expect=expect)

    def test_cluster_show_by_id(self):
        cluster = self.mgr.get(CLUSTER1['id'])
        expect = [('GET', '/v1/clusters/%s' % CLUSTER1['id'], {}, None)]
        self.assertEqual(expect, self.api.calls)
        self.assertEqual(CLUSTER1['name'], cluster.name)
        self.assertEqual(CLUSTER1['cluster_template_id'], cluster.cluster_template_id)

    def test_cluster_show_by_name(self):
        cluster = self.mgr.get(CLUSTER1['name'])
        expect = [('GET', '/v1/clusters/%s' % CLUSTER1['name'], {}, None)]
        self.assertEqual(expect, self.api.calls)
        self.assertEqual(CLUSTER1['name'], cluster.name)
        self.assertEqual(CLUSTER1['cluster_template_id'], cluster.cluster_template_id)

    def test_cluster_create(self):
        cluster = self.mgr.create(**CREATE_CLUSTER)
        expect = [('POST', '/v1/clusters', {}, CREATE_CLUSTER)]
        self.assertEqual(expect, self.api.calls)
        self.assertTrue(cluster)

    def test_cluster_create_with_keypair(self):
        cluster_with_keypair = dict()
        cluster_with_keypair.update(CREATE_CLUSTER)
        cluster_with_keypair['keypair'] = 'test_key'
        cluster = self.mgr.create(**cluster_with_keypair)
        expect = [('POST', '/v1/clusters', {}, cluster_with_keypair)]
        self.assertEqual(expect, self.api.calls)
        self.assertTrue(cluster)

    def test_cluster_create_with_docker_volume_size(self):
        cluster_with_volume_size = dict()
        cluster_with_volume_size.update(CREATE_CLUSTER)
        cluster_with_volume_size['docker_volume_size'] = 20
        cluster = self.mgr.create(**cluster_with_volume_size)
        expect = [('POST', '/v1/clusters', {}, cluster_with_volume_size)]
        self.assertEqual(expect, self.api.calls)
        self.assertTrue(cluster)

    def test_cluster_create_with_labels(self):
        cluster_with_labels = dict()
        cluster_with_labels.update(CREATE_CLUSTER)
        cluster_with_labels['labels'] = 'key=val'
        cluster = self.mgr.create(**cluster_with_labels)
        expect = [('POST', '/v1/clusters', {}, cluster_with_labels)]
        self.assertEqual(expect, self.api.calls)
        self.assertTrue(cluster)

    def test_cluster_create_with_discovery_url(self):
        cluster_with_discovery = dict()
        cluster_with_discovery.update(CREATE_CLUSTER)
        cluster_with_discovery['discovery_url'] = 'discovery_url'
        cluster = self.mgr.create(**cluster_with_discovery)
        expect = [('POST', '/v1/clusters', {}, cluster_with_discovery)]
        self.assertEqual(expect, self.api.calls)
        self.assertTrue(cluster)

    def test_cluster_create_with_cluster_create_timeout(self):
        cluster_with_timeout = dict()
        cluster_with_timeout.update(CREATE_CLUSTER)
        cluster_with_timeout['create_timeout'] = '15'
        cluster = self.mgr.create(**cluster_with_timeout)
        expect = [('POST', '/v1/clusters', {}, cluster_with_timeout)]
        self.assertEqual(expect, self.api.calls)
        self.assertTrue(cluster)

    def test_cluster_create_fail(self):
        CREATE_CLUSTER_FAIL = copy.deepcopy(CREATE_CLUSTER)
        CREATE_CLUSTER_FAIL['wrong_key'] = 'wrong'
        self.assertRaisesRegex(exceptions.InvalidAttribute, 'Key must be in %s' % ','.join(clusters.CREATION_ATTRIBUTES), self.mgr.create, **CREATE_CLUSTER_FAIL)
        self.assertEqual([], self.api.calls)

    def test_cluster_delete_by_id(self):
        cluster = self.mgr.delete(CLUSTER1['id'])
        expect = [('DELETE', '/v1/clusters/%s' % CLUSTER1['id'], {}, None)]
        self.assertEqual(expect, self.api.calls)
        self.assertIsNone(cluster)

    def test_cluster_delete_by_name(self):
        cluster = self.mgr.delete(CLUSTER1['name'])
        expect = [('DELETE', '/v1/clusters/%s' % CLUSTER1['name'], {}, None)]
        self.assertEqual(expect, self.api.calls)
        self.assertIsNone(cluster)

    def test_cluster_update(self):
        patch = {'op': 'replace', 'value': NEW_NAME, 'path': '/name'}
        cluster = self.mgr.update(id=CLUSTER1['id'], patch=patch)
        expect = [('PATCH', '/v1/clusters/%s' % CLUSTER1['id'], {}, patch)]
        self.assertEqual(expect, self.api.calls)
        self.assertEqual(NEW_NAME, cluster.name)

    def test_cluster_update_with_rollback(self):
        patch = {'op': 'replace', 'value': NEW_NAME, 'path': '/name'}
        cluster = self.mgr.update(id=CLUSTER1['id'], patch=patch, rollback=True)
        expect = [('PATCH', '/v1/clusters/%s/?rollback=True' % CLUSTER1['id'], {}, patch)]
        self.assertEqual(expect, self.api.calls)
        self.assertEqual(NEW_NAME, cluster.name)

    def test_cluster_resize(self):
        body = {'node_count': RESIZED_NODE_COUNT}
        cluster = self.mgr.resize(CLUSTER1['uuid'], **body)
        expect = [('POST', '/v1/clusters/%s/actions/resize' % CLUSTER1['uuid'], {}, body)]
        self.assertEqual(expect, self.api.calls)
        self.assertEqual(RESIZED_NODE_COUNT, cluster.node_count)

    def test_cluster_upgrade(self):
        body = {'cluster_template': UPGRADED_TO_TEMPLATE, 'max_batch_size': 1}
        cluster = self.mgr.upgrade(CLUSTER1['uuid'], **body)
        expect = [('POST', '/v1/clusters/%s/actions/upgrade' % CLUSTER1['uuid'], {}, body)]
        self.assertEqual(expect, self.api.calls)
        self.assertEqual(UPGRADED_TO_TEMPLATE, cluster.cluster_template_id)