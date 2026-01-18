from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def AddTierFlag(parser):
    """Adds a --tier flag to the given parser."""
    help_text = "    Tier (or machine type) for this instance, for example: ``db-n1-standard-1''\n    (MySQL instances) or ``db-custom-1-3840'' (PostgreSQL instances). For more\n    information, see\n    [Cloud SQL Instance Settings](https://cloud.google.com/sql/docs/mysql/instance-settings).\n    "
    parser.add_argument('--tier', help=help_text, required=True)