import os
import sys
import subprocess
import time
from pecan.compat import urlopen, URLError
from pecan.tests import PecanTestCase
import unittest
class TestThirdPartyServe(TestTemplateBuilds):

    def poll_http(self, name, proc, port):
        try:
            self.poll(proc)
            retries = 30
            while True:
                retries -= 1
                if retries < 0:
                    raise RuntimeError('The %s server has not replied within 3 seconds.' % name)
                try:
                    resp = urlopen('http://localhost:%d/' % port)
                    assert resp.getcode()
                    assert len(resp.read().decode())
                except URLError:
                    pass
                else:
                    break
                time.sleep(0.1)
        finally:
            proc.terminate()