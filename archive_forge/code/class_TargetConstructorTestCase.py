import testscenarios
import oslo_messaging
from oslo_messaging.tests import utils as test_utils
class TargetConstructorTestCase(test_utils.BaseTestCase):
    scenarios = [('all_none', dict(kwargs=dict())), ('exchange', dict(kwargs=dict(exchange='testexchange'))), ('topic', dict(kwargs=dict(topic='testtopic'))), ('namespace', dict(kwargs=dict(namespace='testnamespace'))), ('version', dict(kwargs=dict(version='3.4'))), ('server', dict(kwargs=dict(server='testserver'))), ('fanout', dict(kwargs=dict(fanout=True)))]

    def test_constructor(self):
        target = oslo_messaging.Target(**self.kwargs)
        for k in self.kwargs:
            self.assertEqual(self.kwargs[k], getattr(target, k))
        for k in ['exchange', 'topic', 'namespace', 'version', 'server', 'fanout']:
            if k in self.kwargs:
                continue
            self.assertIsNone(getattr(target, k))