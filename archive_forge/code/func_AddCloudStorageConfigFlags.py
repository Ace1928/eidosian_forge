from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.pubsub import subscriptions
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.pubsub import resource_args
from googlecloudsdk.command_lib.pubsub import util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
def AddCloudStorageConfigFlags(parser, is_update):
    """Adds Cloud Storage config flags to parser."""
    current_group = parser
    cloud_storage_config_group_help = "Cloud Storage Config Options. The Cloud\n        Pub/Sub service account associated with the enclosing subscription's\n        parent project (i.e.,\n        service-{project_number}@gcp-sa-pubsub.iam.gserviceaccount.com)\n        must have permission to write to this Cloud Storage bucket and to read\n        this bucket's metadata."
    if is_update:
        mutual_exclusive_group = current_group.add_mutually_exclusive_group()
        AddBooleanFlag(parser=mutual_exclusive_group, flag_name='clear-cloud-storage-config', action='store_true', default=None, help_text='If set, clear the Cloud Storage config from the subscription.')
        current_group = mutual_exclusive_group
        cloud_storage_config_group_help += '\n\nNote that an update to the Cloud\n          Storage config will replace it with a new config containing only the\n          flags that are passed in the `update` CLI.'
    cloud_storage_config_group = current_group.add_argument_group(help=cloud_storage_config_group_help)
    cloud_storage_config_group.add_argument('--cloud-storage-bucket', required=True, help='A Cloud Storage bucket to which to write messages for this subscription.')
    cloud_storage_config_group.add_argument('--cloud-storage-file-prefix', default=None, help='The prefix for Cloud Storage filename.')
    cloud_storage_config_group.add_argument('--cloud-storage-file-suffix', default=None, help='The suffix for Cloud Storage filename.')
    cloud_storage_config_group.add_argument('--cloud-storage-file-datetime-format', default=None, help='The custom datetime format string for Cloud Storage filename. See the [datetime format guidance](https://cloud.google.com/pubsub/docs/create-cloudstorage-subscription#file_names).')
    cloud_storage_config_group.add_argument('--cloud-storage-max-bytes', type=arg_parsers.BinarySize(lower_bound='1KB', upper_bound='10GB', default_unit='KB', suggested_binary_size_scales=['KB', 'KiB', 'MB', 'MiB', 'GB', 'GiB']), default=None, help=' The maximum bytes that can be written to a Cloud Storage file before a new file is created. The value must be between 1KB to 10GB. If the unit is omitted, KB is assumed.')
    cloud_storage_config_group.add_argument('--cloud-storage-max-duration', type=arg_parsers.Duration(lower_bound='1m', upper_bound='10m', default_unit='s'), help='The maximum duration that can elapse before a new Cloud Storage\n          file is created. The value must be between 1m and 10m.\n          {}'.format(DURATION_HELP_STR))
    cloud_storage_config_group.add_argument('--cloud-storage-output-format', type=arg_parsers.ArgList(element_type=lambda x: str(x).lower(), min_length=1, max_length=1, choices=['text', 'avro']), default='text', metavar='OUTPUT_FORMAT', help='The output format for data written to Cloud Storage. Values: text (messages will be written as raw text, separated by a newline) or avro (messages will be written as an Avro binary).')
    AddBooleanFlag(parser=cloud_storage_config_group, flag_name='cloud-storage-write-metadata', action='store_true', default=None, help_text='Whether or not to write the subscription name, message_id, publish_time, attributes, and ordering_key as additional fields in the output. The subscription name, message_id, and publish_time fields are put in their own fields while all other message properties other than data (for example, an ordering_key, if present) are added as entries in the attributes map. This has an effect only for subscriptions with --cloud-storage-output-format=avro.')