import sys
import os
import re
from inspect import getfile
import pyomo.common.unittest as unittest
import subprocess
def _failIfPyomoResultsDiffer(self, cmd=None, baseline=None, cwd=None):
    if cwd is None:
        cwd = os.path.dirname(os.path.abspath(getfile(self.__class__)))
    oldpwd = os.getcwd()
    os.chdir(cwd)
    try:
        if os.path.exists(baseline):
            INPUT = open(baseline, 'r')
            baseline = '\n'.join(INPUT.readlines())
            INPUT.close()
        output = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    finally:
        os.chdir(oldpwd)
    if output.returncode != 0:
        self.fail("Command terminated with nonzero status: '%s'" % cmd)
    results = extract_results(re.split('\n', output.stdout))
    try:
        compare_results(results, baseline)
    except IOError:
        err = sys.exc_info()[1]
        self.fail("Command failed to generate results that can be compared with the baseline: '%s'" % err)
    except ValueError:
        err = sys.exc_info()[1]
        self.fail("Difference between results and baseline: '%s'" % err)