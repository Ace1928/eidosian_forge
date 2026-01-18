import unittest
class LoggingResult(_BaseLoggingResult):
    """
    A TestResult implementation which records its method calls.
    """

    def addSubTest(self, test, subtest, err):
        if err is None:
            self._events.append('addSubTestSuccess')
        else:
            self._events.append('addSubTestFailure')
        super().addSubTest(test, subtest, err)