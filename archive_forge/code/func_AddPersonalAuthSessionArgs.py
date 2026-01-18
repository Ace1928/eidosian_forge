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
def AddPersonalAuthSessionArgs(parser):
    """Adds the arguments for enabling personal auth sessions."""
    parser.add_argument('--access-boundary', help='\n        The path to a JSON file specifying the credential access boundary for\n        the personal auth session.\n\n        If not specified, then the access boundary defaults to one that includes\n        the following roles on the containing project:\n\n            roles/storage.objectViewer\n            roles/storage.objectCreator\n            roles/storage.objectAdmin\n            roles/storage.legacyBucketReader\n\n        For more information, see:\n        https://cloud.google.com/iam/docs/downscoping-short-lived-credentials.\n        ')
    parser.add_argument('--openssl-command', hidden=True, help='\n        The full path to the command used to invoke the OpenSSL tool on this\n        machine.\n        ')
    parser.add_argument('--refresh-credentials', action='store_true', default=True, hidden=True, help='\n        Keep the command running to periodically refresh credentials.\n        ')