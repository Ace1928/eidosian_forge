from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
from googlecloudsdk.api_lib.storage import cloud_api
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def add_enable_per_object_retention_flag(parser):
    """Adds flag for enabling object retention for buckets."""
    parser.add_argument('--enable-per-object-retention', action='store_true', help='Enables each object in the bucket to have its own retention settings, which prevents deletion until stored for a specific length of time.')