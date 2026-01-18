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
def ValidateNotificationConfigFlag(args):
    """Raise exception if validation of notification config fails."""
    if 'notification_config' in args._specified_args:
        if 'pubsub' in args.notification_config:
            pubsub = args.notification_config['pubsub']
            if pubsub != 'ENABLED' and pubsub != 'DISABLED':
                raise exceptions.InvalidArgumentException('--notification-config', 'invalid [pubsub] value "{0}"; must be ENABLED or DISABLED.'.format(pubsub))
            if pubsub == 'ENABLED' and 'pubsub-topic' not in args.notification_config:
                raise exceptions.InvalidArgumentException('--notification-config', 'when [pubsub] is ENABLED, [pubsub-topic] must not be empty')
        if 'filter' in args.notification_config:
            known_event_types = ['UpgradeEvent', 'UpgradeAvailableEvent', 'SecurityBulletinEvent']
            lower_known_event_types = []
            for event_type in known_event_types:
                lower_known_event_types.append(event_type.lower())
            filter_opt = args.notification_config['filter']
            inputted_types = filter_opt.split('|')
            for inputted_type in inputted_types:
                if inputted_type.lower() not in lower_known_event_types:
                    raise exceptions.InvalidArgumentException('--notification_config', "valid keys for filter are {0}; received '{1}'".format(known_event_types, inputted_type))