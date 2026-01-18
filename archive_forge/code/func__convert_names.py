import sys
import argparse
import os
import warnings
from . import loader, runner
from .signals import installHandler
def _convert_names(names):
    return [_convert_name(name) for name in names]