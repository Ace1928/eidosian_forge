from functools import wraps
import logging
def _assert_count(self):
    error_msg = '%s != %s' % (self.count, self.expected)
    assert self.count == self.expected, error_msg