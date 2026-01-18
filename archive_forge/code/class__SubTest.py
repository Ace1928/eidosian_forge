import sys
import functools
import difflib
import pprint
import re
import warnings
import collections
import contextlib
import traceback
import types
from . import result
from .util import (strclass, safe_repr, _count_diff_all_purpose,
class _SubTest(TestCase):

    def __init__(self, test_case, message, params):
        super().__init__()
        self._message = message
        self.test_case = test_case
        self.params = params
        self.failureException = test_case.failureException

    def runTest(self):
        raise NotImplementedError('subtests cannot be run directly')

    def _subDescription(self):
        parts = []
        if self._message is not _subtest_msg_sentinel:
            parts.append('[{}]'.format(self._message))
        if self.params:
            params_desc = ', '.join(('{}={!r}'.format(k, v) for k, v in self.params.items()))
            parts.append('({})'.format(params_desc))
        return ' '.join(parts) or '(<subtest>)'

    def id(self):
        return '{} {}'.format(self.test_case.id(), self._subDescription())

    def shortDescription(self):
        """Returns a one-line description of the subtest, or None if no
        description has been provided.
        """
        return self.test_case.shortDescription()

    def __str__(self):
        return '{} {}'.format(self.test_case, self._subDescription())