from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from apitools.base.py import encoding
from googlecloudsdk.calliope import exceptions as calliope_exc
from googlecloudsdk.command_lib.projects import util as projects_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import times
import six
def BuildChannelsFromPrometheusReceivers(receiver_config, messages):
    """Populates a Notification Channel translated from Prometheus alert manager.

  Args:
    receiver_config: Object containing information the Prometheus receiver. For
      example receiver_configs, see
      https://github.com/prometheus/alertmanager/blob/main/doc/examples/simple.yml
    messages: Object containing information about all message types allowed.

  Raises:
    MissingRequiredFieldError: If the provided alert manager file contains
    receivers with missing required field(s).

  Returns:
     The Notification Channel corresponding to the Prometheus alert manager
     provided.
  """
    channels = []
    channel_name = receiver_config.get('name')
    if channel_name is None:
        raise MissingRequiredFieldError('Supplied alert manager file contains receiver without a required field "name"')
    if receiver_config.get('email_configs') is not None:
        for fields in receiver_config.get('email_configs'):
            if fields.get('to') is not None:
                channel = CreateBasePromQLNotificationChannel(channel_name, messages)
                channel.type = 'email'
                channel.labels.additionalProperties.append(messages.NotificationChannel.LabelsValue.AdditionalProperty(key='email_address', value=fields.get('to')))
                channels.append(channel)
    if receiver_config.get('pagerduty_configs') is not None:
        for fields in receiver_config.get('pagerduty_configs'):
            if fields.get('service_key') is not None:
                channel = CreateBasePromQLNotificationChannel(channel_name, messages)
                channel.type = 'pagerduty'
                channel.labels.additionalProperties.append(messages.NotificationChannel.LabelsValue.AdditionalProperty(key='service_key', value=fields.get('service_key')))
                channels.append(channel)
    if receiver_config.get('webhook_configs') is not None:
        for fields in receiver_config.get('webhook_configs'):
            if fields.get('url') is not None:
                channel = CreateBasePromQLNotificationChannel(channel_name, messages)
                channel.type = 'webhook_tokenauth'
                channel.labels.additionalProperties.append(messages.NotificationChannel.LabelsValue.AdditionalProperty(key='url', value=fields.get('url')))
                channels.append(channel)
    supported_receiver_fields = set(['name', 'email_configs', 'pagerduty_configs', 'webhook_configs'])
    for field in receiver_config.keys():
        if field not in supported_receiver_fields:
            log.out.Print('Found unsupported receiver type {field}. {name}.{field} will not be translated.'.format(field=field, name=receiver_config.get('name')))
    return channels