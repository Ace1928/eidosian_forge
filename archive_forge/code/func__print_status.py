import unittest
def _print_status(self, tag, test):
    if self._per_test_output:
        test_id = test.id()
        if test_id.startswith('__main__.'):
            test_id = test_id[len('__main__.'):]
        print('[%s] %s' % (tag, test_id), file=self.stream)
        self.stream.flush()