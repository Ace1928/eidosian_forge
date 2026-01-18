from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import time
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import http_wrapper
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.container import constants as gke_constants
from googlecloudsdk.api_lib.container import util
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources as cloud_resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.util import times
import six
from six.moves import range  # pylint: disable=redefined-builtin
import six.moves.http_client
from cmd argument to set a surge upgrade strategy.
def _GetFilterFromArg(filter_arg, messages):
    """Gets a Filter message object from a filter phrase."""
    if not filter_arg:
        return None
    flag_event_types_to_enum = {'upgradeevent': messages.Filter.EventTypeValueListEntryValuesEnum.UPGRADE_EVENT, 'upgradeavailableevent': messages.Filter.EventTypeValueListEntryValuesEnum.UPGRADE_AVAILABLE_EVENT, 'securitybulletinevent': messages.Filter.EventTypeValueListEntryValuesEnum.SECURITY_BULLETIN_EVENT}
    to_return = messages.Filter()
    for event_type in filter_arg.split('|'):
        event_type = event_type.lower()
        if flag_event_types_to_enum[event_type]:
            to_return.eventType.append(flag_event_types_to_enum[event_type])
    return to_return