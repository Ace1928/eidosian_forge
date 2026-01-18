from itertools import zip_longest
import json
import os
from os.path import abspath, dirname, join
import pyomo.common.unittest as unittest
from pyomo.common.errors import ApplicationError
from pyomo.common.tee import capture_output
from pyomo.dataportal.factory import DataManagerFactory
from pyomo.environ import (
@unittest.skipIf(not yaml_interface, 'YAML interface not available')
class TestYamlPortal(TestTextPortal):
    suffix = '.yaml'
    skiplist = ['tableD', 'tableT', 'tableU', 'tableXW_4']

    def create_options(self, name):
        return {'filename': os.path.abspath(tutorial_dir + os.sep + 'yaml' + os.sep + name + self.suffix)}