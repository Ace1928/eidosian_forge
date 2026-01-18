import copy
import gyp.input
import argparse
import os.path
import re
import shlex
import sys
import traceback
from gyp.common import GypError
def Noop(value):
    return value