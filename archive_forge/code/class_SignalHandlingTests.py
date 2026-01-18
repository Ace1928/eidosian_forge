import os
import signal
import time
from http.client import BadStatusLine
import pytest
import portend
import cherrypy
import cherrypy.process.servers
from cherrypy.test import helper
class SignalHandlingTests(helper.CPWebCase):

    def test_SIGHUP_tty(self):
        try:
            from signal import SIGHUP
        except ImportError:
            return self.skip('skipped (no SIGHUP) ')
        p = helper.CPProcess(ssl=self.scheme.lower() == 'https')
        p.write_conf(extra='test_case_name: "test_SIGHUP_tty"')
        p.start(imports='cherrypy.test._test_states_demo')
        os.kill(p.get_pid(), SIGHUP)
        p.join()

    def test_SIGHUP_daemonized(self):
        try:
            from signal import SIGHUP
        except ImportError:
            return self.skip('skipped (no SIGHUP) ')
        if os.name not in ['posix']:
            return self.skip('skipped (not on posix) ')
        p = helper.CPProcess(ssl=self.scheme.lower() == 'https', wait=True, daemonize=True)
        p.write_conf(extra='test_case_name: "test_SIGHUP_daemonized"')
        p.start(imports='cherrypy.test._test_states_demo')
        pid = p.get_pid()
        try:
            os.kill(pid, SIGHUP)
            time.sleep(2)
            self.getPage('/pid')
            self.assertStatus(200)
            new_pid = int(self.body)
            self.assertNotEqual(new_pid, pid)
        finally:
            self.getPage('/exit')
        p.join()

    def _require_signal_and_kill(self, signal_name):
        if not hasattr(signal, signal_name):
            self.skip('skipped (no %(signal_name)s)' % vars())
        if not hasattr(os, 'kill'):
            self.skip('skipped (no os.kill)')

    def test_SIGTERM(self):
        """SIGTERM should shut down the server whether daemonized or not."""
        self._require_signal_and_kill('SIGTERM')
        p = helper.CPProcess(ssl=self.scheme.lower() == 'https')
        p.write_conf(extra='test_case_name: "test_SIGTERM"')
        p.start(imports='cherrypy.test._test_states_demo')
        os.kill(p.get_pid(), signal.SIGTERM)
        p.join()
        if os.name in ['posix']:
            p = helper.CPProcess(ssl=self.scheme.lower() == 'https', wait=True, daemonize=True)
            p.write_conf(extra='test_case_name: "test_SIGTERM_2"')
            p.start(imports='cherrypy.test._test_states_demo')
            os.kill(p.get_pid(), signal.SIGTERM)
            p.join()

    def test_signal_handler_unsubscribe(self):
        self._require_signal_and_kill('SIGTERM')
        if os.name == 'nt':
            self.skip('SIGTERM not available')
        p = helper.CPProcess(ssl=self.scheme.lower() == 'https')
        p.write_conf(extra='unsubsig: True\ntest_case_name: "test_signal_handler_unsubscribe"\n')
        p.start(imports='cherrypy.test._test_states_demo')
        os.kill(p.get_pid(), signal.SIGTERM)
        p.join()
        with open(p.error_log, 'rb') as f:
            log_lines = list(f)
            assert any((line.endswith(b'I am an old SIGTERM handler.\n') for line in log_lines))