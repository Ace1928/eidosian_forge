from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.dataplex import util as dataplex_api
from googlecloudsdk.command_lib.iam import iam_util
def GenerateEnvironmentForCreateRequest(args):
    """Create Environment for Message Create Requests."""
    module = dataplex_api.GetMessageModule()
    request = module.GoogleCloudDataplexV1Environment(description=args.description, displayName=args.display_name, labels=dataplex_api.CreateLabels(module.GoogleCloudDataplexV1Environment, args), infrastructureSpec=GenerateInfrastructureSpec(args), sessionSpec=GenerateSessionSpec(args))
    return request