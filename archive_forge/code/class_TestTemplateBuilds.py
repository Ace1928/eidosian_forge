import os
import sys
import subprocess
import time
from pecan.compat import urlopen, URLError
from pecan.tests import PecanTestCase
import unittest
class TestTemplateBuilds(PecanTestCase):
    """
        Used to test the templated quickstart project(s).
        """

    @property
    def bin(self):
        return os.path.dirname(sys.executable)

    def poll(self, proc):
        limit = 30
        for i in range(limit):
            proc.poll()
            if proc.returncode is None:
                break
            elif i == limit:
                raise RuntimeError("Server process didn't start.")
            time.sleep(0.1)

    def test_project_pecan_serve_command(self):
        proc = subprocess.Popen([os.path.join(self.bin, 'pecan'), 'serve', 'testing123/config.py'])
        try:
            self.poll(proc)
            retries = 30
            while True:
                retries -= 1
                if retries < 0:
                    raise RuntimeError('The HTTP server has not replied within 3 seconds.')
                try:
                    resp = urlopen('http://localhost:8080/')
                    assert resp.getcode()
                    assert len(resp.read().decode())
                except URLError:
                    pass
                else:
                    break
                time.sleep(0.1)
        finally:
            proc.terminate()

    def test_project_pecan_shell_command(self):
        proc = subprocess.Popen([os.path.join(self.bin, 'pecan'), 'shell', 'testing123/config.py'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE)
        self.poll(proc)
        out, _ = proc.communicate(b'{"model" : model, "conf" : conf, "app" : app}')
        assert 'testing123.model' in out.decode(), out
        assert 'Config(' in out.decode(), out
        assert 'webtest.app.TestApp' in out.decode(), out
        try:
            proc.terminate()
        except:
            pass