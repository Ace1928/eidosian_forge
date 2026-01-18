from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
def AddIndexFileFlag(parser):
    """Adds a index_file flag to the given parser.

  Args:
    parser: The argparse parser.
  """
    parser.add_argument('index_file', help='\n        The path to your `index.yaml` file. For a detailed look into defining\n        your `index.yaml` file, refer to this configuration guide:\n        https://cloud.google.com/datastore/docs/tools/indexconfig#Datastore_About_index_yaml\n        ')