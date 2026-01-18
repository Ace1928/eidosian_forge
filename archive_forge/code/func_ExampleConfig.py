import argparse
import enum
import os
import os.path
import pickle
import re
import sys
import types
import pyomo.common.unittest as unittest
from io import StringIO
from pyomo.common.dependencies import yaml, yaml_available, yaml_load_args
from pyomo.common.config import (
from pyomo.common.log import LoggingIntercept
def ExampleConfig():
    CONFIG = ConfigDict()
    CONFIG.declare('option_1', ConfigValue(default=5, domain=int, doc='The first configuration option'))
    SOLVER = CONFIG.declare('solver_options', ConfigDict())
    SOLVER.declare('solver_option_1', ConfigValue(default=1, domain=float, doc='The first solver configuration option', visibility=DEVELOPER_OPTION))
    SOLVER.declare('solver_option_2', ConfigValue(default=1, domain=float, doc='The second solver configuration option\n\n        With a very long line containing\n        wrappable text in a long, silly paragraph\n        with little actual information.\n        #) but a bulleted list\n        #) with two bullets\n        '))
    SOLVER.declare('solver_option_3', ConfigValue(default=1, domain=float, doc='\n            The third solver configuration option\n\n            This has a leading newline and a very long line containing\n            wrappable text in a long, silly paragraph with\n            little actual information.\n\n         .. and_a_list::\n            #) but a bulleted list\n            #) with two bullets '))
    CONFIG.declare('option_2', ConfigValue(default=5, domain=int, doc='The second solver configuration option\n        with a very long line containing\n        wrappable text in a long, silly paragraph\n        with little actual information.\n        '))
    return CONFIG