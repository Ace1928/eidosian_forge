import os
import json
import os.path
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.scripting.pyomo_main import main
from pyomo.scripting.util import cleanup
from pyomo.neos.kestrel import kestrelAMPL
import pyomo.neos
import pyomo.environ as pyo
from pyomo.common.fileutils import this_file_dir
@unittest.pytest.mark.neos
@unittest.skipIf(not neos_available, 'Cannot make connection to NEOS server')
@unittest.skipUnless(email_set, 'NEOS_EMAIL not set')
class TestSolvers_direct_call_min(RunAllNEOSSolvers, DirectDriver, unittest.TestCase):
    sense = pyo.minimize