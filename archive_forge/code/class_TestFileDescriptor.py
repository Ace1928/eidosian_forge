import os
import time
import sys
from io import StringIO, BytesIO
from pyomo.common.log import LoggingIntercept
import pyomo.common.unittest as unittest
from pyomo.common.tempfiles import TempfileManager
import pyomo.common.tee as tee
class TestFileDescriptor(unittest.TestCase):

    def setUp(self):
        self.out = sys.stdout
        self.out_fd = os.dup(1)

    def tearDown(self):
        sys.stdout = self.out
        os.dup2(self.out_fd, 1)
        os.close(self.out_fd)

    def _generate_output(self, redirector):
        with redirector:
            sys.stdout.write('to_stdout_1\n')
            sys.stdout.flush()
            with os.fdopen(1, 'w', closefd=False) as F:
                F.write('to_fd1_1\n')
                F.flush()
        sys.stdout.write('to_stdout_2\n')
        sys.stdout.flush()
        with os.fdopen(1, 'w', closefd=False) as F:
            F.write('to_fd1_2\n')
            F.flush()

    def test_redirect_synchronize_stdout(self):
        r, w = os.pipe()
        os.dup2(w, 1)
        sys.stdout = os.fdopen(1, 'w', closefd=False)
        rd = tee.redirect_fd(synchronize=True)
        self._generate_output(rd)
        with os.fdopen(r, 'r') as FILE:
            os.close(w)
            os.close(1)
            self.assertEqual(FILE.read(), 'to_stdout_2\nto_fd1_2\n')

    def test_redirect_no_synchronize_stdout(self):
        r, w = os.pipe()
        os.dup2(w, 1)
        sys.stdout = os.fdopen(1, 'w', closefd=False)
        rd = tee.redirect_fd(synchronize=False)
        self._generate_output(rd)
        with os.fdopen(r, 'r') as FILE:
            os.close(w)
            os.close(1)
            self.assertEqual(FILE.read(), 'to_stdout_1\nto_stdout_2\nto_fd1_2\n')

    @unittest.pytest.fixture(autouse=True)
    def capfd(self, capfd):
        """
        Reimplementation needed for use in unittest.TestCase subclasses
        """
        self.capfd = capfd

    def test_redirect_synchronize_stdout_not_fd1(self):
        self.capfd.disabled()
        r, w = os.pipe()
        os.dup2(w, 1)
        rd = tee.redirect_fd(synchronize=True)
        self._generate_output(rd)
        with os.fdopen(r, 'r') as FILE:
            os.close(w)
            os.close(1)
            self.assertEqual(FILE.read(), 'to_fd1_2\n')

    def test_redirect_no_synchronize_stdout_not_fd1(self):
        self.capfd.disabled()
        r, w = os.pipe()
        os.dup2(w, 1)
        rd = tee.redirect_fd(synchronize=False)
        self._generate_output(rd)
        with os.fdopen(r, 'r') as FILE:
            os.close(w)
            os.close(1)
            self.assertEqual(FILE.read(), 'to_fd1_2\n')

    def test_redirect_synchronize_stringio(self):
        r, w = os.pipe()
        os.dup2(w, 1)
        try:
            sys.stdout, out = (StringIO(), sys.stdout)
            rd = tee.redirect_fd(synchronize=True)
            self._generate_output(rd)
        finally:
            sys.stdout, out = (out, sys.stdout)
        self.assertEqual(out.getvalue(), 'to_stdout_2\n')
        with os.fdopen(r, 'r') as FILE:
            os.close(w)
            os.close(1)
            self.assertEqual(FILE.read(), 'to_fd1_2\n')

    def test_redirect_no_synchronize_stringio(self):
        r, w = os.pipe()
        os.dup2(w, 1)
        try:
            sys.stdout, out = (StringIO(), sys.stdout)
            rd = tee.redirect_fd(synchronize=False)
            self._generate_output(rd)
        finally:
            sys.stdout, out = (out, sys.stdout)
        self.assertEqual(out.getvalue(), 'to_stdout_1\nto_stdout_2\n')
        with os.fdopen(r, 'r') as FILE:
            os.close(w)
            os.close(1)
            self.assertEqual(FILE.read(), 'to_fd1_2\n')

    def test_capture_output_fd(self):
        r, w = os.pipe()
        os.dup2(w, 1)
        sys.stdout = os.fdopen(1, 'w', closefd=False)
        with tee.capture_output(capture_fd=True) as OUT:
            sys.stdout.write('to_stdout_1\n')
            sys.stdout.flush()
            with os.fdopen(1, 'w', closefd=False) as F:
                F.write('to_fd1_1\n')
                F.flush()
        sys.stdout.write('to_stdout_2\n')
        sys.stdout.flush()
        with os.fdopen(1, 'w', closefd=False) as F:
            F.write('to_fd1_2\n')
            F.flush()
        self.assertEqual(OUT.getvalue(), 'to_stdout_1\nto_fd1_1\n')
        with os.fdopen(r, 'r') as FILE:
            os.close(1)
            os.close(w)
            self.assertEqual(FILE.read(), 'to_stdout_2\nto_fd1_2\n')