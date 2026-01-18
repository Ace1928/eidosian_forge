import unittest
def addSubTest(self, test, subtest, err):
    if err is None:
        self._events.append('addSubTestSuccess')
    else:
        self._events.append('addSubTestFailure')
    super().addSubTest(test, subtest, err)