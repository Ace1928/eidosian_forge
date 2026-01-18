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
def _validateTemplate(self, config, reference_template, **kwds):
    test = config.generate_yaml_template(**kwds)
    width = kwds.get('width', 80)
    indent = kwds.get('indent_spacing', 2)
    sys.stdout.write(test)
    for l in test.splitlines():
        self.assertLessEqual(len(l), width)
        if l.strip().startswith('#'):
            continue
        self.assertEqual((len(l) - len(l.lstrip())) % indent, 0)
    self.assertEqual(test, reference_template)