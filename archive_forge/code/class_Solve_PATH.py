import os
import pyomo.common.unittest as unittest
from pyomo.common.dependencies import yaml, yaml_available, yaml_load_args
from pyomo.common.fileutils import this_file_dir, PYOMO_ROOT_DIR
import pyomo.opt
import pyomo.scripting.pyomo_main as pyomo_main
from pyomo.scripting.util import cleanup
import pyomo.environ
@unittest.skipIf(not yaml_available, 'YAML is not available')
@unittest.skipIf(not 'path' in solvers, "The 'path' executable is not available")
class Solve_PATH(unittest.TestCase, CommonTests):

    def tearDown(self):
        if os.path.exists(os.path.join(currdir, 'result.yml')):
            os.remove(os.path.join(currdir, 'result.yml'))