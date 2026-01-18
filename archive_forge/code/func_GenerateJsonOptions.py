from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.dataplex import util as dataplex_api
from googlecloudsdk.command_lib.iam import iam_util
def GenerateJsonOptions(args):
    return dataplex_api.GetMessageModule().GoogleCloudDataplexV1ZoneDiscoverySpecJsonOptions(encoding=args.json_encoding, disableTypeInference=args.json_disable_type_inference)