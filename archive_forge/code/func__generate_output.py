import os
import time
import sys
from io import StringIO, BytesIO
from pyomo.common.log import LoggingIntercept
import pyomo.common.unittest as unittest
from pyomo.common.tempfiles import TempfileManager
import pyomo.common.tee as tee
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