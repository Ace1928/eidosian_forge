import logging
from unittest import mock
import fixtures
from oslotest import base as test_base
from oslo_utils import excutils
from oslo_utils import timeutils
class SaveAndReraiseTest(test_base.BaseTestCase):

    def test_save_and_reraise_exception_forced(self):

        def _force_reraise():
            try:
                raise IOError('I broke')
            except Exception:
                with excutils.save_and_reraise_exception() as e:
                    e.reraise = False
                e.force_reraise()
        self.assertRaises(IOError, _force_reraise)

    def test_save_and_reraise_exception_capture_reraise(self):

        def _force_reraise():
            try:
                raise IOError('I broke')
            except Exception:
                excutils.save_and_reraise_exception().capture().force_reraise()
        self.assertRaises(IOError, _force_reraise)

    def test_save_and_reraise_exception_capture_not_active(self):
        e = excutils.save_and_reraise_exception()
        self.assertRaises(RuntimeError, e.capture, check=True)

    def test_save_and_reraise_exception_forced_not_active(self):
        e = excutils.save_and_reraise_exception()
        self.assertRaises(RuntimeError, e.force_reraise)
        e = excutils.save_and_reraise_exception()
        e.capture(check=False)
        self.assertRaises(RuntimeError, e.force_reraise)

    def test_save_and_reraise_exception(self):
        e = None
        msg = 'foo'
        try:
            try:
                raise Exception(msg)
            except Exception:
                with excutils.save_and_reraise_exception():
                    pass
        except Exception as _e:
            e = _e
        self.assertEqual(str(e), msg)

    @mock.patch('logging.getLogger')
    def test_save_and_reraise_exception_dropped(self, get_logger_mock):
        logger = get_logger_mock()
        e = None
        msg = 'second exception'
        try:
            try:
                raise Exception('dropped')
            except Exception:
                with excutils.save_and_reraise_exception():
                    raise Exception(msg)
        except Exception as _e:
            e = _e
        self.assertEqual(str(e), msg)
        self.assertTrue(logger.error.called)

    def test_save_and_reraise_exception_no_reraise(self):
        """Test that suppressing the reraise works."""
        try:
            raise Exception('foo')
        except Exception:
            with excutils.save_and_reraise_exception() as ctxt:
                ctxt.reraise = False

    @mock.patch('logging.getLogger')
    def test_save_and_reraise_exception_dropped_no_reraise(self, get_logger_mock):
        logger = get_logger_mock()
        e = None
        msg = 'second exception'
        try:
            try:
                raise Exception('dropped')
            except Exception:
                with excutils.save_and_reraise_exception(reraise=False):
                    raise Exception(msg)
        except Exception as _e:
            e = _e
        self.assertEqual(str(e), msg)
        self.assertFalse(logger.error.called)

    def test_save_and_reraise_exception_provided_logger(self):
        fake_logger = mock.MagicMock()
        try:
            try:
                raise Exception('foo')
            except Exception:
                with excutils.save_and_reraise_exception(logger=fake_logger):
                    raise Exception('second exception')
        except Exception:
            pass
        self.assertTrue(fake_logger.error.called)