from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.compute import metadata_utils
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import completers as compute_completers
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.instances import flags as instances_flags
from googlecloudsdk.command_lib.compute.resource_policies import flags as maintenance_flags
from googlecloudsdk.command_lib.util.args import labels_util
def AddBulkCreateArgs(parser, add_zone_region_flags, support_max_count_per_zone, support_custom_hostnames):
    """Adds bulk creation specific arguments to parser."""
    parser.add_argument('--count', type=int, help='\n      Number of Compute Engine virtual machines to create. If specified, and\n      `--predefined-names` is specified, count must equal the amount of names\n      provided to `--predefined-names`. If not specified,\n      the number of virtual machines created will equal the number of names\n      provided to `--predefined-names`.\n    ')
    parser.add_argument('--min-count', type=int, help='\n        The minimum number of Compute Engine virtual machines that must be\n        successfully created for the operation to be considered a success. If\n        the operation successfully creates as many virtual machines as\n        specified here they will be persisted, otherwise the operation rolls\n        back and deletes all created virtual machines. If not specified, this\n        value is equal to `--count`.')
    name_group = parser.add_group(mutex=True, required=True)
    name_group.add_argument('--predefined-names', type=arg_parsers.ArgList(), metavar='INSTANCE_NAME', help='\n        List of predefined names for the Compute Engine virtual machines being\n        created. If `--count` is specified alongside this flag, provided count\n        must equal the amount of names provided to this flag. If `--count` is\n        not specified, the number of virtual machines\n        created will equal the number of names provided.\n      ')
    name_group.add_argument('--name-pattern', help="\n        Name pattern for generating instance names. Specify a pattern with a\n        single sequence of hash (#) characters that will be replaced with\n        generated sequential numbers of instances. E.g. name pattern of\n        'instance-###' will generate instance names 'instance-001',\n        'instance-002', and so on, until the number of virtual machines\n        specified using `--count` is reached. If instances matching name pattern\n        exist, the new instances will be assigned names to avoid clashing with\n        the existing ones. E.g. if there exists `instance-123`, the new\n        instances will start at `instance-124` and increment from there.\n      ")
    if add_zone_region_flags:
        location = parser.add_group(required=True, mutex=True)
        location.add_argument('--region', help='\n        Region in which to create the Compute Engine virtual machines. Compute\n        Engine will select a zone in which to create all virtual machines.\n    ')
        location.add_argument('--zone', help='\n        Zone in which to create the Compute Engine virtual machines.\n\n        A list of zones can be fetched by running:\n\n            $ gcloud compute zones list\n\n        To unset the property, run:\n\n            $ gcloud config unset compute/zone\n\n        Alternatively, the zone can be stored in the environment variable\n        CLOUDSDK_COMPUTE_ZONE.\n     ')
    parser.add_argument('--location-policy', metavar='ZONE=POLICY', type=arg_parsers.ArgDict(), help='\n        Policy for which zones to include or exclude during bulk instance creation\n        within a region. Policy is defined as a list of key-value pairs, with the\n        key being the zone name, and value being the applied policy. Available\n        policies are `allow` and `deny`. Default for zones if left unspecified is `allow`.\n\n        Example:\n\n          gcloud compute instances bulk create --name-pattern=example-###\n            --count=5 --region=us-east1\n            --location-policy=us-east1-b=allow,us-east1-c=deny\n      ')
    if support_max_count_per_zone:
        parser.add_argument('--max-count-per-zone', metavar='ZONE=MAX_COUNT_PER_ZONE', type=arg_parsers.ArgDict(), help='\n          Maximum number of instances per zone specified as key-value pairs. The zone name is the key and the max count per zone\n          is the value in that zone.\n\n          Example:\n\n            gcloud compute instances bulk create --name-pattern=example-###\n              --count=5 --region=us-east1\n              --max-count-per-zone=us-east1-b=2,us-east-1-c=1\n        ')
    if support_custom_hostnames:
        parser.add_argument('--per-instance-hostnames', metavar='INSTANCE_NAME=INSTANCE_HOSTNAME', type=arg_parsers.ArgDict(key_type=str, value_type=str), help='\n          Specify the hostname of the instance to be created. The specified\n          hostname must be RFC1035 compliant. If hostname is not specified, the\n          default hostname is [INSTANCE_NAME].c.[PROJECT_ID].internal when using\n          the global DNS, and [INSTANCE_NAME].[ZONE].c.[PROJECT_ID].internal\n          when using zonal DNS.\n        ')