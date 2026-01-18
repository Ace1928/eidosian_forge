import functools
import os
import unittest
def _repeat_with_success_at_least(f):

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        assert len(args) > 0
        instance = args[0]
        assert isinstance(instance, unittest.TestCase)
        success_counter = 0
        failure_counter = 0
        results = []

        def fail():
            msg = '\nFail: {0}, Success: {1}'.format(failure_counter, success_counter)
            if len(results) > 0:
                first = results[0]
                errs = first.failures + first.errors
                if len(errs) > 0:
                    err_msg = '\n'.join((fail[1] for fail in errs))
                    msg += '\n\nThe first error message:\n' + err_msg
            instance.fail(msg)
        for _ in range(times):
            suite = unittest.TestSuite()
            ins = type(instance)(instance._testMethodName)
            suite.addTest(unittest.FunctionTestCase(lambda: f(ins, *args[1:], **kwargs), setUp=ins.setUp, tearDown=ins.tearDown))
            result = QuietTestRunner().run(suite)
            if len(result.skipped) == 1:
                instance.skipTest(result.skipped[0][1])
            elif result.wasSuccessful():
                success_counter += 1
            else:
                results.append(result)
                failure_counter += 1
            if success_counter >= min_success:
                instance.assertTrue(True)
                return
            if failure_counter > times - min_success:
                fail()
                return
        fail()
    return wrapper