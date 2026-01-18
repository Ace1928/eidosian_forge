import copy
import gyp.input
import argparse
import os.path
import re
import shlex
import sys
import traceback
from gyp.common import GypError
def RegenerateAppendFlag(flag, values, predicate, env_name, options):
    """Regenerate a list of command line flags, for an option of action='append'.

  The |env_name|, if given, is checked in the environment and used to generate
  an initial list of options, then the options that were specified on the
  command line (given in |values|) are appended.  This matches the handling of
  environment variables and command line flags where command line flags override
  the environment, while not requiring the environment to be set when the flags
  are used again.
  """
    flags = []
    if options.use_environment and env_name:
        for flag_value in ShlexEnv(env_name):
            value = FormatOpt(flag, predicate(flag_value))
            if value in flags:
                flags.remove(value)
            flags.append(value)
    if values:
        for flag_value in values:
            flags.append(FormatOpt(flag, predicate(flag_value)))
    return flags