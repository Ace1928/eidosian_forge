from oslo_utils import uuidutils
from taskflow.engines.action_engine import executor
from taskflow.engines.worker_based import protocol as pr
from taskflow import exceptions as excp
from taskflow import test
from taskflow.test import mock
from taskflow.tests import utils
from taskflow.types import failure
class TestProtocolValidation(test.TestCase):

    def test_send_notify(self):
        msg = pr.Notify()
        pr.Notify.validate(msg.to_dict(), False)

    def test_send_notify_invalid(self):
        msg = {'all your base': 'are belong to us'}
        self.assertRaises(excp.InvalidFormat, pr.Notify.validate, msg, False)

    def test_reply_notify(self):
        msg = pr.Notify(topic='bob', tasks=['a', 'b', 'c'])
        pr.Notify.validate(msg.to_dict(), True)

    def test_reply_notify_invalid(self):
        msg = {'topic': {}, 'tasks': 'not yours'}
        self.assertRaises(excp.InvalidFormat, pr.Notify.validate, msg, True)

    def test_request(self):
        request = pr.Request(utils.DummyTask('hi'), uuidutils.generate_uuid(), pr.EXECUTE, {}, 1.0)
        pr.Request.validate(request.to_dict())

    def test_request_invalid(self):
        msg = {'task_name': 1, 'task_cls': False, 'arguments': []}
        self.assertRaises(excp.InvalidFormat, pr.Request.validate, msg)

    def test_request_invalid_action(self):
        request = pr.Request(utils.DummyTask('hi'), uuidutils.generate_uuid(), pr.EXECUTE, {}, 1.0)
        request = request.to_dict()
        request['action'] = 'NOTHING'
        self.assertRaises(excp.InvalidFormat, pr.Request.validate, request)

    def test_response_progress(self):
        msg = pr.Response(pr.EVENT, details={'progress': 0.5}, event_type='blah')
        pr.Response.validate(msg.to_dict())

    def test_response_completion(self):
        msg = pr.Response(pr.SUCCESS, result=1)
        pr.Response.validate(msg.to_dict())

    def test_response_mixed_invalid(self):
        msg = pr.Response(pr.EVENT, details={'progress': 0.5}, event_type='blah', result=1)
        self.assertRaises(excp.InvalidFormat, pr.Response.validate, msg)

    def test_response_bad_state(self):
        msg = pr.Response('STUFF')
        self.assertRaises(excp.InvalidFormat, pr.Response.validate, msg)