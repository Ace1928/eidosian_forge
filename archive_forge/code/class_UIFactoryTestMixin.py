import unittest
from ... import tests, transport, ui
from ..ui_testing import StringIOAsTTY, StringIOWithEncoding, TextUIFactory
class UIFactoryTestMixin:
    """Common tests for UIFactories.

    These are supposed to be expressed with no assumptions about how the
    UIFactory implements the method, only that it does implement them (or
    fails cleanly), and that the concrete subclass will make arrangements to
    build a factory and to examine its behaviour.

    Note that this is *not* a TestCase, because it can't be directly run, but
    the concrete subclasses should be.
    """

    def test_be_quiet(self):
        self.factory.be_quiet(True)
        self.assertEqual(True, self.factory.is_quiet())
        self.factory.be_quiet(False)
        self.assertEqual(False, self.factory.is_quiet())

    def test_confirm_action(self):
        self._load_responses([True])
        result = self.factory.confirm_action('Break a lock?', 'bzr.lock.break.confirm', {})
        self.assertEqual(result, True)

    def test_note(self):
        self.factory.note('a note to the user')
        self._check_note('a note to the user')

    def test_show_error(self):
        msg = 'an error occurred'
        self.factory.show_error(msg)
        self._check_show_error(msg)

    def test_show_message(self):
        msg = 'a message'
        self.factory.show_message(msg)
        self._check_show_message(msg)

    def test_show_warning(self):
        msg = 'a warning'
        self.factory.show_warning(msg)
        self._check_show_warning(msg)

    def test_make_output_stream(self):
        output_stream = self.factory.make_output_stream()
        output_stream.write('hello!')

    def test_transport_activity(self):
        t = transport.get_transport_from_url('memory:///')
        self.factory.report_transport_activity(t, 1000, 'write')
        self.factory.report_transport_activity(t, 2000, 'read')
        self.factory.report_transport_activity(t, 4000, None)
        self.factory.log_transport_activity()
        self._check_log_transport_activity_noarg()
        self.factory.log_transport_activity(display=True)
        self._check_log_transport_activity_display()

    def test_no_transport_activity(self):
        t = transport.get_transport_from_url('memory:///')
        self.factory.log_transport_activity(display=True)
        self._check_log_transport_activity_display_no_bytes()