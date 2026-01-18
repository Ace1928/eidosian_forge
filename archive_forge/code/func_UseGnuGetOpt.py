from mx import DateTime
from __future__ import absolute_import
from __future__ import print_function
import os
import pdb
import sys
import traceback
from absl import app
from absl import flags
def UseGnuGetOpt(choice=True):
    """Allow mixed flag/arg ordering in subcommand argv."""
    global _cmd_gnugetopt
    _cmd_gnugetopt = choice