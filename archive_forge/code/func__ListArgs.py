from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.services import apikeys
from googlecloudsdk.calliope import base
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def _ListArgs(parser):
    parser.add_argument('--show-deleted', action='store_true', help='Show soft-deleted keys by specifying this flag.')