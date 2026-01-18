from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.database_migration import api_util
def AddAlloydbClusterFlag(parser, required=False):
    """Adds the --alloydb-cluster flag to the given parser."""
    help_text = '    If the destination is an AlloyDB cluster, use this field to provide the\n    AlloyDB cluster ID.\n  '
    parser.add_argument('--alloydb-cluster', help=help_text, required=required)