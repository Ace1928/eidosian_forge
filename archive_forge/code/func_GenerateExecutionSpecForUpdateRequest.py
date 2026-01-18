from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.dataplex import util as dataplex_api
from googlecloudsdk.api_lib.util import messages as messages_util
from googlecloudsdk.command_lib.iam import iam_util
def GenerateExecutionSpecForUpdateRequest(args):
    """Generate ExecutionSpec From Arguments."""
    module = dataplex_api.GetMessageModule()
    executionspec = module.GoogleCloudDataplexV1DataScanExecutionSpec(trigger=GenerateTrigger(args))
    return executionspec