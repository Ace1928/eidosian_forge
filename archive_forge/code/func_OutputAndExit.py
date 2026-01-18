from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
import sys
import warnings
def OutputAndExit(message):
    sys.stderr.write('%s\n' % message)
    sys.exit(1)