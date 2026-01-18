from unittest import mock
from urllib import parse as urlparse
from heat.common import template_format
from heat.engine.clients import client_plugin
from heat.engine import resource
from heat.engine.resources.openstack.zaqar import queue
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
class ZaqarSignedQueueURLTest(common.HeatTestCase):
    tmpl = '\nheat_template_version: 2015-10-15\nresources:\n  signed_url:\n    type: OS::Zaqar::SignedQueueURL\n    properties:\n      queue: foo\n      ttl: 60\n      paths:\n        - messages\n        - subscription\n      methods:\n        - POST\n        - DELETE\n'

    @mock.patch('zaqarclient.queues.v2.queues.Queue.signed_url')
    def test_create(self, mock_signed_url):
        mock_signed_url.return_value = {'expires': '2020-01-01', 'signature': 'secret', 'project': 'project_id', 'paths': ['/v2/foo/messages', '/v2/foo/sub'], 'methods': ['DELETE', 'POST']}
        self.t = template_format.parse(self.tmpl)
        self.stack = utils.parse_stack(self.t)
        self.rsrc = self.stack['signed_url']
        self.assertIsNone(self.rsrc.validate())
        self.stack.create()
        self.assertEqual(self.rsrc.CREATE, self.rsrc.action)
        self.assertEqual(self.rsrc.COMPLETE, self.rsrc.status)
        self.assertEqual(self.stack.CREATE, self.stack.action)
        self.assertEqual(self.stack.COMPLETE, self.stack.status)
        mock_signed_url.assert_called_once_with(paths=['messages', 'subscription'], methods=['POST', 'DELETE'], ttl_seconds=60)
        self.assertEqual('secret', self.rsrc.FnGetAtt('signature'))
        self.assertEqual('2020-01-01', self.rsrc.FnGetAtt('expires'))
        self.assertEqual('project_id', self.rsrc.FnGetAtt('project'))
        self.assertEqual(['/v2/foo/messages', '/v2/foo/sub'], self.rsrc.FnGetAtt('paths'))
        self.assertEqual(['DELETE', 'POST'], self.rsrc.FnGetAtt('methods'))
        expected_query = {'queue_name': ['foo'], 'expires': ['2020-01-01'], 'signature': ['secret'], 'project_id': ['project_id'], 'paths': ['/v2/foo/messages,/v2/foo/sub'], 'methods': ['DELETE,POST']}
        query_str_attr = self.rsrc.get_attribute('query_str')
        self.assertEqual(expected_query, urlparse.parse_qs(query_str_attr, strict_parsing=True))