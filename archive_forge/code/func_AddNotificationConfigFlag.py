from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import textwrap
from googlecloudsdk.api_lib.compute import constants as compute_constants
from googlecloudsdk.api_lib.container import api_adapter
from googlecloudsdk.api_lib.container import util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.container import constants
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from Google Kubernetes Engine labels that are used for the purpose of tracking
from the node pool, depending on whether locations are being added or removed.
def AddNotificationConfigFlag(parser, hidden=False):
    """Adds a --notification-config flag to the given parser."""
    help_text = 'The notification configuration of the cluster. GKE supports publishing\ncluster upgrade notifications to any Pub/Sub topic you created in the same\nproject. Create a subscription for the topic specified to receive notification\nmessages. See https://cloud.google.com/pubsub/docs/admin on how to manage\nPub/Sub topics and subscriptions. You can also use the filter option to\nspecify which event types you\'d like to receive from the following options:\nSecurityBulletinEvent, UpgradeEvent, UpgradeAvailableEvent.\n\nExamples:\n\n  $ {command} example-cluster --notification-config=pubsub=ENABLED,pubsub-topic=projects/{project}/topics/{topic-name}\n  $ {command} example-cluster --notification-config=pubsub=ENABLED,pubsub-topic=projects/{project}/topics/{topic-name},filter="SecurityBulletinEvent|UpgradeEvent"\n\nThe project of the Pub/Sub topic must be the same one as the cluster. It can\nbe either the project ID or the project number.\n'
    return parser.add_argument('--notification-config', type=arg_parsers.ArgDict(spec={'pubsub': str, 'pubsub-topic': str, 'filter': str}, required_keys=['pubsub']), metavar='pubsub=ENABLED|DISABLED,pubsub-topic=TOPIC', help=help_text, hidden=hidden)