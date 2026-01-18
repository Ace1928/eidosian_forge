import datetime
import operator
import warnings
import pytest
import tempfile
import re
import sys
import numpy as np
from numpy.testing import (
from numpy.core._multiarray_tests import fromstring_null_term_c_api
class _DeprecationTestCase:
    message = ''
    warning_cls = DeprecationWarning

    def setup_method(self):
        self.warn_ctx = warnings.catch_warnings(record=True)
        self.log = self.warn_ctx.__enter__()
        warnings.filterwarnings('always', category=self.warning_cls)
        warnings.filterwarnings('always', message=self.message, category=self.warning_cls)

    def teardown_method(self):
        self.warn_ctx.__exit__()

    def assert_deprecated(self, function, num=1, ignore_others=False, function_fails=False, exceptions=np._NoValue, args=(), kwargs={}):
        """Test if DeprecationWarnings are given and raised.

        This first checks if the function when called gives `num`
        DeprecationWarnings, after that it tries to raise these
        DeprecationWarnings and compares them with `exceptions`.
        The exceptions can be different for cases where this code path
        is simply not anticipated and the exception is replaced.

        Parameters
        ----------
        function : callable
            The function to test
        num : int
            Number of DeprecationWarnings to expect. This should normally be 1.
        ignore_others : bool
            Whether warnings of the wrong type should be ignored (note that
            the message is not checked)
        function_fails : bool
            If the function would normally fail, setting this will check for
            warnings inside a try/except block.
        exceptions : Exception or tuple of Exceptions
            Exception to expect when turning the warnings into an error.
            The default checks for DeprecationWarnings. If exceptions is
            empty the function is expected to run successfully.
        args : tuple
            Arguments for `function`
        kwargs : dict
            Keyword arguments for `function`
        """
        __tracebackhide__ = True
        self.log[:] = []
        if exceptions is np._NoValue:
            exceptions = (self.warning_cls,)
        try:
            function(*args, **kwargs)
        except Exception if function_fails else tuple():
            pass
        num_found = 0
        for warning in self.log:
            if warning.category is self.warning_cls:
                num_found += 1
            elif not ignore_others:
                raise AssertionError('expected %s but got: %s' % (self.warning_cls.__name__, warning.category))
        if num is not None and num_found != num:
            msg = '%i warnings found but %i expected.' % (len(self.log), num)
            lst = [str(w) for w in self.log]
            raise AssertionError('\n'.join([msg] + lst))
        with warnings.catch_warnings():
            warnings.filterwarnings('error', message=self.message, category=self.warning_cls)
            try:
                function(*args, **kwargs)
                if exceptions != tuple():
                    raise AssertionError('No error raised during function call')
            except exceptions:
                if exceptions == tuple():
                    raise AssertionError('Error raised during function call')

    def assert_not_deprecated(self, function, args=(), kwargs={}):
        """Test that warnings are not raised.

        This is just a shorthand for:

        self.assert_deprecated(function, num=0, ignore_others=True,
                        exceptions=tuple(), args=args, kwargs=kwargs)
        """
        self.assert_deprecated(function, num=0, ignore_others=True, exceptions=tuple(), args=args, kwargs=kwargs)