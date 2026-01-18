from __future__ import print_function
import argparse
import sys
import graphviz
from ._discover import findMachines
def _gvhtml(s):
    return '<{}>'.format(s)