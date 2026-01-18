from __future__ import nested_scopes
import fnmatch
import os.path
from _pydev_runfiles.pydev_runfiles_coverage import start_coverage_support
from _pydevd_bundle.pydevd_constants import *  # @UnusedWildImport
import re
import time
class DjangoTestSuiteRunner:

    def __init__(self):
        pass

    def run_tests(self, *args, **kwargs):
        raise AssertionError("Unable to run suite with django.test.runner.DiscoverRunner nor django.test.simple.DjangoTestSuiteRunner because it couldn't be imported.")