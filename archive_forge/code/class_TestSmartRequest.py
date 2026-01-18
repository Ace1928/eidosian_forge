import threading
from breezy import errors, transport
from breezy.bzr.bzrdir import BzrDir
from breezy.bzr.smart import request
from breezy.tests import TestCase, TestCaseWithMemoryTransport
class TestSmartRequest(TestCase):

    def test_request_class_without_do_body(self):
        """If a request has no body data, and the request's implementation does
        not override do_body, then no exception is raised.
        """
        handler = request.SmartServerRequestHandler(None, {b'foo': NoBodyRequest}, '/')
        handler.args_received((b'foo',))
        handler.end_received()

    def test_only_request_code_is_jailed(self):
        transport = 'dummy transport'
        handler = request.SmartServerRequestHandler(transport, {b'foo': CheckJailRequest}, '/')
        handler.args_received((b'foo',))
        self.assertEqual(None, request.jail_info.transports)
        handler.accept_body(b'bytes')
        self.assertEqual(None, request.jail_info.transports)
        handler.end_received()
        self.assertEqual(None, request.jail_info.transports)
        self.assertEqual([[transport]] * 3, handler._command.jail_transports_log)

    def test_all_registered_requests_are_safety_qualified(self):
        unclassified_requests = []
        allowed_info = ('read', 'idem', 'mutate', 'semivfs', 'semi', 'stream')
        for key in request.request_handlers.keys():
            info = request.request_handlers.get_info(key)
            if info is None or info not in allowed_info:
                unclassified_requests.append(key)
        if unclassified_requests:
            self.fail('These requests were not categorized as safe/unsafe to retry: %s' % (unclassified_requests,))