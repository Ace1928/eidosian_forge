import sys
import argparse
import os
import warnings
from . import loader, runner
from .signals import installHandler
def _convert_select_pattern(pattern):
    if not '*' in pattern:
        pattern = '*%s*' % pattern
    return pattern