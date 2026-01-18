from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.dataplex import util as dataplex_api
from googlecloudsdk.api_lib.util import messages as messages_util
from googlecloudsdk.command_lib.iam import iam_util
def GenerateData(args):
    """Generate Data From Arguments."""
    module = dataplex_api.GetMessageModule()
    if args.IsSpecified('data_source_entity'):
        data = module.GoogleCloudDataplexV1DataSource(entity=args.data_source_entity)
    else:
        data = module.GoogleCloudDataplexV1DataSource(resource=args.data_source_resource)
    return data