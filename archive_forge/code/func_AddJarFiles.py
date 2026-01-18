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
def AddJarFiles(parser):
    """Add --jars flag."""
    parser.add_argument('--jars', type=arg_parsers.ArgList(), metavar='JAR', default=[], help='Comma-separated list of jar files to be provided to the classpaths.')