import logging
from unittest import mock
import fixtures
from oslotest import base as test_base
from oslo_utils import excutils
from oslo_utils import timeutils
class ExceptionFilterTest(test_base.BaseTestCase):

    def _make_filter_func(self, ignore_classes=AssertionError):

        @excutils.exception_filter
        def ignore_exceptions(ex):
            """Ignore some exceptions F."""
            return isinstance(ex, ignore_classes)
        return ignore_exceptions

    def _make_filter_method(self, ignore_classes=AssertionError):

        class ExceptionIgnorer(object):

            def __init__(self, ignore):
                self.ignore = ignore

            @excutils.exception_filter
            def ignore_exceptions(self, ex):
                """Ignore some exceptions M."""
                return isinstance(ex, self.ignore)
        return ExceptionIgnorer(ignore_classes).ignore_exceptions

    def _make_filter_classmethod(self, ignore_classes=AssertionError):

        class ExceptionIgnorer(object):
            ignore = ignore_classes

            @excutils.exception_filter
            @classmethod
            def ignore_exceptions(cls, ex):
                """Ignore some exceptions C."""
                return isinstance(ex, cls.ignore)
        return ExceptionIgnorer.ignore_exceptions

    def _make_filter_staticmethod(self, ignore_classes=AssertionError):

        class ExceptionIgnorer(object):

            @excutils.exception_filter
            @staticmethod
            def ignore_exceptions(ex):
                """Ignore some exceptions S."""
                return isinstance(ex, ignore_classes)
        return ExceptionIgnorer.ignore_exceptions

    def test_filter_func_call(self):
        ignore_assertion_error = self._make_filter_func()
        try:
            assert False, 'This is a test'
        except Exception as exc:
            ignore_assertion_error(exc)

    def test_raise_func_call(self):
        ignore_assertion_error = self._make_filter_func()
        try:
            raise RuntimeError
        except Exception as exc:
            self.assertRaises(RuntimeError, ignore_assertion_error, exc)

    def test_raise_previous_func_call(self):
        ignore_assertion_error = self._make_filter_func()
        try:
            raise RuntimeError
        except Exception as exc1:
            try:
                raise RuntimeError
            except Exception as exc2:
                self.assertIsNot(exc1, exc2)
            raised = self.assertRaises(RuntimeError, ignore_assertion_error, exc1)
            self.assertIs(exc1, raised)

    def test_raise_previous_after_filtered_func_call(self):
        ignore_assertion_error = self._make_filter_func()
        try:
            raise RuntimeError
        except Exception as exc1:
            try:
                assert False, 'This is a test'
            except Exception:
                pass
            self.assertRaises(RuntimeError, ignore_assertion_error, exc1)

    def test_raise_other_func_call(self):

        @excutils.exception_filter
        def translate_exceptions(ex):
            raise RuntimeError
        try:
            assert False, 'This is a test'
        except Exception as exc:
            self.assertRaises(RuntimeError, translate_exceptions, exc)

    def test_filter_func_context_manager(self):
        ignore_assertion_error = self._make_filter_func()
        with ignore_assertion_error:
            assert False, 'This is a test'

    def test_raise_func_context_manager(self):
        ignore_assertion_error = self._make_filter_func()

        def try_runtime_err():
            with ignore_assertion_error:
                raise RuntimeError
        self.assertRaises(RuntimeError, try_runtime_err)

    def test_raise_other_func_context_manager(self):

        @excutils.exception_filter
        def translate_exceptions(ex):
            raise RuntimeError

        def try_assertion():
            with translate_exceptions:
                assert False, 'This is a test'
        self.assertRaises(RuntimeError, try_assertion)

    def test_noexc_func_context_manager(self):
        ignore_assertion_error = self._make_filter_func()
        with ignore_assertion_error:
            pass

    def test_noexc_nocall_func_context_manager(self):

        @excutils.exception_filter
        def translate_exceptions(ex):
            raise RuntimeError
        with translate_exceptions:
            pass

    def test_func_docstring(self):
        ignore_func = self._make_filter_func()
        self.assertEqual('Ignore some exceptions F.', ignore_func.__doc__)

    def test_filter_method_call(self):
        ignore_assertion_error = self._make_filter_method()
        try:
            assert False, 'This is a test'
        except Exception as exc:
            ignore_assertion_error(exc)

    def test_raise_method_call(self):
        ignore_assertion_error = self._make_filter_method()
        try:
            raise RuntimeError
        except Exception as exc:
            self.assertRaises(RuntimeError, ignore_assertion_error, exc)

    def test_filter_method_context_manager(self):
        ignore_assertion_error = self._make_filter_method()
        with ignore_assertion_error:
            assert False, 'This is a test'

    def test_raise_method_context_manager(self):
        ignore_assertion_error = self._make_filter_method()

        def try_runtime_err():
            with ignore_assertion_error:
                raise RuntimeError
        self.assertRaises(RuntimeError, try_runtime_err)

    def test_method_docstring(self):
        ignore_func = self._make_filter_method()
        self.assertEqual('Ignore some exceptions M.', ignore_func.__doc__)

    def test_filter_classmethod_call(self):
        ignore_assertion_error = self._make_filter_classmethod()
        try:
            assert False, 'This is a test'
        except Exception as exc:
            ignore_assertion_error(exc)

    def test_raise_classmethod_call(self):
        ignore_assertion_error = self._make_filter_classmethod()
        try:
            raise RuntimeError
        except Exception as exc:
            self.assertRaises(RuntimeError, ignore_assertion_error, exc)

    def test_filter_classmethod_context_manager(self):
        ignore_assertion_error = self._make_filter_classmethod()
        with ignore_assertion_error:
            assert False, 'This is a test'

    def test_raise_classmethod_context_manager(self):
        ignore_assertion_error = self._make_filter_classmethod()

        def try_runtime_err():
            with ignore_assertion_error:
                raise RuntimeError
        self.assertRaises(RuntimeError, try_runtime_err)

    def test_classmethod_docstring(self):
        ignore_func = self._make_filter_classmethod()
        self.assertEqual('Ignore some exceptions C.', ignore_func.__doc__)

    def test_filter_staticmethod_call(self):
        ignore_assertion_error = self._make_filter_staticmethod()
        try:
            assert False, 'This is a test'
        except Exception as exc:
            ignore_assertion_error(exc)

    def test_raise_staticmethod_call(self):
        ignore_assertion_error = self._make_filter_staticmethod()
        try:
            raise RuntimeError
        except Exception as exc:
            self.assertRaises(RuntimeError, ignore_assertion_error, exc)

    def test_filter_staticmethod_context_manager(self):
        ignore_assertion_error = self._make_filter_staticmethod()
        with ignore_assertion_error:
            assert False, 'This is a test'

    def test_raise_staticmethod_context_manager(self):
        ignore_assertion_error = self._make_filter_staticmethod()

        def try_runtime_err():
            with ignore_assertion_error:
                raise RuntimeError
        self.assertRaises(RuntimeError, try_runtime_err)

    def test_staticmethod_docstring(self):
        ignore_func = self._make_filter_staticmethod()
        self.assertEqual('Ignore some exceptions S.', ignore_func.__doc__)