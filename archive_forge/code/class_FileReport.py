from __future__ import with_statement
import inspect
import keyword
import os
import re
import sys
import time
import tokenize
import warnings
from fnmatch import fnmatch
from optparse import OptionParser
class FileReport(BaseReport):
    """Collect the results of the checks and print only the filenames."""
    print_filename = True