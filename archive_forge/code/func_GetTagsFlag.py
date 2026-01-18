from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.core.util import files
def GetTagsFlag():
    return base.Argument('--tags', required=False, type=arg_parsers.ArgDict(), metavar='TAG=VALUE', help='Tags for the package.')