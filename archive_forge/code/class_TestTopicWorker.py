from oslo_utils import reflection
from taskflow.engines.worker_based import types as worker_types
from taskflow import test
from taskflow.test import mock
from taskflow.tests import utils
class TestTopicWorker(test.TestCase):

    def test_topic_worker(self):
        worker = worker_types.TopicWorker('dummy-topic', [utils.DummyTask], identity='dummy')
        self.assertTrue(worker.performs(utils.DummyTask))
        self.assertFalse(worker.performs(utils.NastyTask))
        self.assertEqual('dummy', worker.identity)
        self.assertEqual('dummy-topic', worker.topic)