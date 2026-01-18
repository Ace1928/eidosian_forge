import testscenarios
import oslo_messaging
from oslo_messaging.tests import utils as test_utils
class TargetReprTestCase(test_utils.BaseTestCase):
    scenarios = [('all_none', dict(kwargs=dict(), repr='')), ('exchange', dict(kwargs=dict(exchange='testexchange'), repr='exchange=testexchange')), ('topic', dict(kwargs=dict(topic='testtopic'), repr='topic=testtopic')), ('namespace', dict(kwargs=dict(namespace='testnamespace'), repr='namespace=testnamespace')), ('version', dict(kwargs=dict(version='3.4'), repr='version=3.4')), ('server', dict(kwargs=dict(server='testserver'), repr='server=testserver')), ('fanout', dict(kwargs=dict(fanout=True), repr='fanout=True')), ('exchange_and_fanout', dict(kwargs=dict(exchange='testexchange', fanout=True), repr='exchange=testexchange, fanout=True'))]

    def test_repr(self):
        target = oslo_messaging.Target(**self.kwargs)
        self.assertEqual('<Target ' + self.repr + '>', str(target))