from unittest import mock
import uuid
from oslotest import base as test_base
import statsd
import webob.dec
import webob.exc
from oslo_middleware import stats
class TestStaticMethods(test_base.BaseTestCase):

    def test_removes_uuid(self):
        id = str(uuid.uuid4())
        path = 'foo.{uuid}.bar'.format(uuid=id)
        stat = stats.StatsMiddleware.strip_uuid(path)
        self.assertEqual('foo.bar', stat)

    def test_removes_short_uuid(self):
        id = uuid.uuid4().hex
        path = 'foo.{uuid}.bar'.format(uuid=id)
        stat = stats.StatsMiddleware.strip_short_uuid(path)
        self.assertEqual('foo.bar', stat)

    def test_strips_dots_from_version(self):
        path = '/v1.2/foo.bar/bar.foo'
        stat = stats.StatsMiddleware.strip_dot_from_version(path)
        self.assertEqual('/v12/foo.bar/bar.foo', stat)