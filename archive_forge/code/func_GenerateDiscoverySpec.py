from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.dataplex import util as dataplex_api
from googlecloudsdk.command_lib.iam import iam_util
def GenerateDiscoverySpec(args):
    return dataplex_api.GetMessageModule().GoogleCloudDataplexV1ZoneDiscoverySpec(enabled=args.discovery_enabled, includePatterns=args.discovery_include_patterns, excludePatterns=args.discovery_exclude_patterns, schedule=args.discovery_schedule, csvOptions=GenerateCsvOptions(args), jsonOptions=GenerateJsonOptions(args))