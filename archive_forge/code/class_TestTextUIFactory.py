import unittest
from ... import tests, transport, ui
from ..ui_testing import StringIOAsTTY, StringIOWithEncoding, TextUIFactory
class TestTextUIFactory(tests.TestCase, UIFactoryTestMixin):

    def setUp(self):
        super().setUp()
        self.factory = self._create_ui_factory()
        self.factory.__enter__()
        self.addCleanup(self.factory.__exit__, None, None, None)
        self.stdin = self.factory.stdin
        self.stdout = self.factory.stdout
        self.stderr = self.factory.stderr

    def _create_ui_factory(self):
        return TextUIFactory('')

    def _check_note(self, note_text):
        self.assertEqual('%s\n' % note_text, self.stdout.getvalue())

    def _check_show_error(self, msg):
        self.assertEqual('bzr: error: %s\n' % msg, self.stderr.getvalue())
        self.assertEqual('', self.stdout.getvalue())

    def _check_show_message(self, msg):
        self.assertEqual('%s\n' % msg, self.stdout.getvalue())
        self.assertEqual('', self.stderr.getvalue())

    def _check_show_warning(self, msg):
        self.assertEqual('bzr: warning: %s\n' % msg, self.stderr.getvalue())
        self.assertEqual('', self.stdout.getvalue())

    def _check_log_transport_activity_noarg(self):
        self.assertEqual('', self.stdout.getvalue())
        self.assertContainsRe(self.stderr.getvalue(), '\\d+kB\\s+\\dkB/s |')
        self.assertNotContainsRe(self.stderr.getvalue(), 'Transferred:')

    def _check_log_transport_activity_display(self):
        self.assertEqual('', self.stdout.getvalue())
        self.assertEqual('', self.stderr.getvalue())

    def _check_log_transport_activity_display_no_bytes(self):
        self.assertEqual('', self.stdout.getvalue())
        self.assertEqual('', self.stderr.getvalue())

    def _load_responses(self, responses):
        self.factory.stdin.seek(0)
        self.factory.stdin.writelines([r and 'y\n' or 'n\n' for r in responses])
        self.factory.stdin.seek(0)