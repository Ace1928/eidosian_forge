from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.container import api_adapter as container_api_adapter
from googlecloudsdk.api_lib.krmapihosting import util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import log
def AsyncLog(operation):
    """Print log messages for async commands."""
    log.status.Print('\n      Check operation [{}] for status.\n      To describe the operation, run:\n\n        $ gcloud anthos config operations describe {}'.format(operation.name, operation.name))
    return operation