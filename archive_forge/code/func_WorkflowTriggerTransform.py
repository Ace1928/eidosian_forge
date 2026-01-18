from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from typing import MutableMapping
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_exceptions
from googlecloudsdk.core import yaml
def WorkflowTriggerTransform(trigger, resources):
    """Transform workflow trigger according to the proto.

  Refer to go/gcb-v2-filters to understand more details.

  Args:
    trigger: the trigger defined in the workflow YAML.
    resources: the workflow resources dictionary.
  Raises:
    InvalidYamlError: The eventType was unsupported.
  """
    trigger['id'] = trigger.pop('name')
    event_source = trigger.pop('source', trigger.pop('eventSource', ''))
    if event_source:
        if event_source not in resources:
            raise cloudbuild_exceptions.InvalidYamlError('Unfound event source: {event_source} in workflow resources'.format(event_source=event_source))
        if 'secret' in resources[event_source]:
            trigger['webhookSecret'] = {'id': event_source}
        else:
            trigger['source'] = {'id': event_source}
    if 'secret' in trigger:
        secret = trigger.pop('secret')
        if secret not in resources:
            raise cloudbuild_exceptions.InvalidYamlError('Unfound secret: {secret} in workflow resources'.format(secret=secret))
        trigger['webhookSecret'] = {'id': secret}
    event_type_mapping = {'branch-push': 'PUSH_BRANCH', 'tag-push': 'PUSH_TAG', 'pull-request': 'PULL_REQUEST', 'any': 'ALL'}
    if 'eventType' in trigger:
        event_type = trigger.pop('eventType')
        mapped_event_type = event_type_mapping.get(event_type)
        if mapped_event_type is not None:
            trigger['eventType'] = mapped_event_type
        else:
            raise cloudbuild_exceptions.InvalidYamlError('Unsupported event type: {event_type}. Supported: {event_types}'.format(event_type=event_type, event_types=','.join(event_type_mapping.keys())))
    for key, value in trigger.pop('filters', {}).items():
        trigger[key] = value
    if 'gitRef' in trigger and 'regex' in trigger['gitRef']:
        trigger['gitRef']['nameRegex'] = trigger['gitRef'].pop('regex')
    ParamDictTransform(trigger.get('params', []))