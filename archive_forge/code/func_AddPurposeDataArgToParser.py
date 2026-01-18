from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def AddPurposeDataArgToParser(parser):
    """Adds argument for the TagKey's purpose data to the parser.

  Args:
     parser: ArgumentInterceptor, An argparse parser.
  """
    parser.add_argument('--purpose-data', type=arg_parsers.ArgDict(spec={'network': str}, max_length=1), help='Purpose data of the TagKey that can only be set on creation. This data is validated by the policy system that corresponds to the purpose.')