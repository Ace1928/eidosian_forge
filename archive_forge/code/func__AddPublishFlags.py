from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.pubsub import lite_util
from googlecloudsdk.command_lib.util.args import resource_args
def _AddPublishFlags(parser):
    """Adds publish arguments to the parser."""
    resource_args.AddResourceArgToParser(parser, resource_path='pubsub.lite_topic', help_text='The pubsub lite topic to publish to.', required=True)
    parser.add_argument('--message', help='The body of the message to publish to the given topic name.')
    parser.add_argument('--attributes', metavar='KEY=VALUE', type=arg_parsers.ArgDict(key_type=str, value_type=str, max_length=100), help='Comma-separated list of attributes. Each ATTRIBUTE has the form\n          name="value". You can specify up to 100 attributes.')
    parser.add_argument('--ordering-key', help='A string key, used for ordering delivery to subscribers. All\n          messages with the same ordering key will be assigned to the same\n          partition for ordered delivery.')
    parser.add_argument('--event-time', type=arg_parsers.Datetime.Parse, metavar='DATETIME', help='A user-specified event time. Run `gcloud topic datetimes` for\n          information on time formats.')