import sys
import warnings
from functools import partial
from textwrap import indent
import pytest
from nibabel.deprecator import (
from ..testing import clear_and_catch_warnings
class TestDeprecatorFunc:
    """Test deprecator function specified in ``dep_func``"""
    dep_func = Deprecator(cmp_func)

    def test_dep_func(self):
        dec = self.dep_func
        func = dec('foo')(func_no_doc)
        with pytest.deprecated_call():
            assert func() is None
        assert func.__doc__ == 'foo\n'
        func = dec('foo')(func_doc)
        with pytest.deprecated_call() as w:
            assert func(1) is None
            assert len(w) == 1
        assert func.__doc__ == 'A docstring\n\nfoo\n'
        func = dec('foo')(func_doc_long)
        with pytest.deprecated_call() as w:
            assert func(1, 2) is None
            assert len(w) == 1
        assert func.__doc__ == f'A docstring\n   \n   foo\n   \n{indent(TESTSETUP, '   ', lambda x: True)}   Some text\n{indent(TESTCLEANUP, '   ', lambda x: True)}'
        func = dec('foo', '1.1')(func_no_doc)
        assert func.__doc__ == 'foo\n\n* deprecated from version: 1.1\n'
        with pytest.deprecated_call() as w:
            assert func() is None
            assert len(w) == 1
        func = dec('foo', until='99.4')(func_no_doc)
        with pytest.deprecated_call() as w:
            assert func() is None
            assert len(w) == 1
        assert func.__doc__ == f'foo\n\n* Will raise {ExpiredDeprecationError} as of version: 99.4\n'
        func = dec('foo', until='1.8')(func_no_doc)
        with pytest.raises(ExpiredDeprecationError):
            func()
        assert func.__doc__ == f'foo\n\n* Raises {ExpiredDeprecationError} as of version: 1.8\n'
        func = dec('foo', '1.2', '1.8')(func_no_doc)
        with pytest.raises(ExpiredDeprecationError):
            func()
        assert func.__doc__ == f'foo\n\n* deprecated from version: 1.2\n* Raises {ExpiredDeprecationError} as of version: 1.8\n'
        func = dec('foo', '1.2', '1.8')(func_doc_long)
        assert func.__doc__ == f'A docstring\n\nfoo\n\n* deprecated from version: 1.2\n* Raises {ExpiredDeprecationError} as of version: 1.8\n'
        with pytest.raises(ExpiredDeprecationError):
            func()
        func = dec('foo', warn_class=UserWarning)(func_no_doc)
        with clear_and_catch_warnings(modules=[_OWN_MODULE]) as w:
            warnings.simplefilter('always')
            assert func() is None
            assert len(w) == 1
            assert w[0].category is UserWarning
        func = dec('foo', error_class=CustomError)(func_no_doc)
        with pytest.deprecated_call():
            assert func() is None
        func = dec('foo', until='1.8', error_class=CustomError)(func_no_doc)
        with pytest.raises(CustomError):
            func()