import unittest
from ... import tests, transport, ui
from ..ui_testing import StringIOAsTTY, StringIOWithEncoding, TextUIFactory
class TestTTYTextUIFactory(TestTextUIFactory):

    def _create_ui_factory(self):
        self.overrideEnv('TERM', None)
        return TextUIFactory('', StringIOAsTTY(), StringIOAsTTY())

    def _check_log_transport_activity_display(self):
        self.assertEqual('', self.stdout.getvalue())
        self.assertContainsRe(self.stderr.getvalue(), 'Transferred: 7kB \\(\\d+\\.\\dkB/s r:2kB w:1kB u:4kB\\)')

    def _check_log_transport_activity_display_no_bytes(self):
        self.assertEqual('', self.stdout.getvalue())
        self.assertEqual('', self.stderr.getvalue())