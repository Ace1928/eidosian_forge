from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import datetime
import time
import uuid
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.apis import yaml_data
from googlecloudsdk.command_lib.util.args import resource_args
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import times
def ProcessTableCopyConfiguration(ref, args, request):
    """Build JobConfigurationTableCopy from request resource args."""
    del ref
    source_ref = args.CONCEPTS.source.Parse()
    destination_ref = args.CONCEPTS.destination.Parse()
    arg_utils.SetFieldInMessage(request, 'job.configuration.copy.destinationTable.datasetId', destination_ref.Parent().Name())
    arg_utils.SetFieldInMessage(request, 'job.configuration.copy.destinationTable.projectId', destination_ref.projectId)
    arg_utils.SetFieldInMessage(request, 'job.configuration.copy.destinationTable.tableId', destination_ref.Name())
    arg_utils.SetFieldInMessage(request, 'job.configuration.copy.sourceTable.datasetId', source_ref.Parent().Name())
    arg_utils.SetFieldInMessage(request, 'job.configuration.copy.sourceTable.projectId', source_ref.projectId)
    arg_utils.SetFieldInMessage(request, 'job.configuration.copy.sourceTable.tableId', source_ref.Name())
    return request