import unittest
import webtest
from wsme import WSRoot, expose, validate
from wsme.rest import scan_api
from wsme import types
from wsme import exc
import wsme.api as wsme_api
import wsme.types
from wsme.tests.test_protocols import DummyProtocol
class TestFunctionDefinition(unittest.TestCase):

    def test_get_arg(self):

        def myfunc(self):
            pass
        fd = wsme_api.FunctionDefinition(wsme_api.FunctionDefinition)
        fd.arguments.append(wsme_api.FunctionArgument('a', int, True, None))
        assert fd.get_arg('a').datatype is int
        assert fd.get_arg('b') is None