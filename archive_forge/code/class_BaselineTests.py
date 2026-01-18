from itertools import zip_longest
import json
import re
import glob
import subprocess
import os
from os.path import join
import pyomo.common.unittest as unittest
from pyomo.common.dependencies import attempt_import
from pyomo.common.fileutils import this_file_dir
from pyomo.common.tempfiles import TempfileManager
import pyomo.scripting.pyomo_main as main
class BaselineTests(Tests):

    @parameterized.parameterized.expand(input=names)
    def nlwriter_baseline_test(self, name):
        baseline = join(currdir, name + '.pyomo.nl')
        testFile = TempfileManager.create_tempfile(suffix=name + '.test.nl')
        cmd = ['--output=' + testFile, join(currdir, name + '_testCase.py')]
        if os.path.exists(join(currdir, name + '.dat')):
            cmd.append(join(currdir, name + '.dat'))
        self.pyomo(cmd)
        with open(testFile, 'r') as f1, open(baseline, 'r') as f2:
            f1_contents = list(filter(None, f1.read().replace('n', 'n ').split()))
            f2_contents = list(filter(None, f2.read().replace('n', 'n ').split()))
            for item1, item2 in zip_longest(f1_contents, f2_contents):
                try:
                    self.assertEqual(float(item1), float(item2))
                except:
                    self.assertEqual(item1, item2)