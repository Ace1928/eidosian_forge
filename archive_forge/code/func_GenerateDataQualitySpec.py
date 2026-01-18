from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.dataplex import util as dataplex_api
from googlecloudsdk.api_lib.util import messages as messages_util
from googlecloudsdk.command_lib.iam import iam_util
def GenerateDataQualitySpec(args):
    """Generate DataQualitySpec From Arguments."""
    module = dataplex_api.GetMessageModule()
    if args.IsSpecified('data_quality_spec_file'):
        dataqualityspec = dataplex_api.ReadObject(args.data_quality_spec_file)
        if dataqualityspec is not None:
            dataqualityspec = messages_util.DictToMessageWithErrorCheck(dataplex_api.SnakeToCamelDict(dataqualityspec), module.GoogleCloudDataplexV1DataQualitySpec)
    else:
        dataqualityspec = module.GoogleCloudDataplexV1DataQualitySpec()
    return dataqualityspec