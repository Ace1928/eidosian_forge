from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
import json
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
import six
def AddPythonFiles(parser):
    """Add --py-files flag."""
    parser.add_argument('--py-files', type=arg_parsers.ArgList(), metavar='PY', default=[], help="Comma-separated list of Python scripts to be passed to the PySpark framework. Supported file types: ``.py'', ``.egg'' and ``.zip.''")