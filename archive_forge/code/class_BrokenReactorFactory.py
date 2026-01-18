from hamcrest import assert_that, equal_to, has_length
from typing_extensions import NoReturn
from twisted.trial._dist.test.matchers import matches_result
from twisted.trial.reporter import TestResult
from twisted.trial.runner import TestLoader
from twisted.trial.unittest import SynchronousTestCase, TestSuite
from .reactormixins import ReactorBuilder
class BrokenReactorFactory(ReactorBuilder, SynchronousTestCase):
    _reactors = ['twisted.internet.test.test_reactormixins.unsupportedReactor']

    def test_brokenFactory(self) -> None:
        """
                Try, and fail, to build an unsupported reactor.
                """
        self.buildReactor()