from hamcrest import assert_that, equal_to, has_length
from typing_extensions import NoReturn
from twisted.trial._dist.test.matchers import matches_result
from twisted.trial.reporter import TestResult
from twisted.trial.runner import TestLoader
from twisted.trial.unittest import SynchronousTestCase, TestSuite
from .reactormixins import ReactorBuilder
class ReactorBuilderTests(SynchronousTestCase):
    """
    Tests for L{ReactorBuilder}.
    """

    def test_buildReactorFails(self) -> None:
        """
        If the reactor factory raises any exception then
        L{ReactorBuilder.buildReactor} raises L{SkipTest}.
        """

        class BrokenReactorFactory(ReactorBuilder, SynchronousTestCase):
            _reactors = ['twisted.internet.test.test_reactormixins.unsupportedReactor']

            def test_brokenFactory(self) -> None:
                """
                Try, and fail, to build an unsupported reactor.
                """
                self.buildReactor()
        cases = BrokenReactorFactory.makeTestCaseClasses().values()
        loader = TestLoader()
        suite = TestSuite((loader.loadClass(cls) for cls in cases))
        result = TestResult()
        suite.run(result)
        assert_that(result, matches_result(skips=has_length(1)))
        [(_, skip)] = result.skips
        assert_that(skip, equal_to(UNSUPPORTED))