from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import sys
import textwrap
from googlecloudsdk.api_lib.artifacts import exceptions as ar_exceptions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import properties
def GetJsonKeyFlag(tool):
    """Gets Json Key Flag text based on specified tool."""
    if tool == 'pypi' or tool == 'python':
        return base.Argument('--json-key', help='Path to service account JSON key. If not specified, output returns either credentials for an active service account or a placeholder for the current user account.')
    elif tool in ('gradle', 'maven', 'npm'):
        return base.Argument('--json-key', help='Path to service account JSON key. If not specified, current active service account credentials or a placeholder for gcloud credentials is used.')
    else:
        raise ar_exceptions.ArtifactRegistryError('Invalid tool type: {}'.format(tool))