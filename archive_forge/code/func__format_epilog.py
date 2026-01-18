import argparse
import fnmatch
import importlib
import inspect
import re
import sys
from docutils import nodes
from docutils.parsers import rst
from docutils.parsers.rst import directives
from docutils import statemachine
from cliff import app
from cliff import commandmanager
def _format_epilog(parser):
    """Get parser epilog.

    We parse this as reStructuredText, allowing users to embed rich
    information in their help messages if they so choose.
    """
    for line in statemachine.string2lines(parser.epilog, tab_width=4, convert_whitespace=True):
        yield line