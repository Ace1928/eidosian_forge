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
def PrometheusMessageFromString(rule_yaml, messages, channels):
    """Populates Alert Policies translated from Prometheus alert rules.

  Args:
    rule_yaml: Opened object of the Prometheus YAML file provided.
    messages: Object containing information about all message types allowed.
    channels: List of Notification Channel names to be added to the translated
      policies.

  Raises:
    YamlOrJsonLoadError: If the YAML file cannot be loaded.

  Returns:
     A list of the Alert Policies corresponding to the Prometheus rules YAML
     file provided.
  """
    try:
        contents = yaml.load(rule_yaml)
        if contents is None:
            raise ValueError('Failed to load YAML file. Is it empty?')
        policies = []
        if contents.get('groups') is None:
            raise ValueError('No groups')
        for group in contents.get('groups'):
            if group.get('rules') is None:
                raise ValueError('No rules in group "%s"' % group.get('name'))
            for rule in group.get('rules'):
                condition = BuildPrometheusCondition(messages, group, rule)
                policy = messages.AlertPolicy()
                policy.conditions.append(condition)
                if rule.get('annotations') is not None:
                    policy.documentation = messages.Documentation()
                    if rule.get('annotations').get('subject') is not None:
                        policy.documentation.subject = TranslatePromQLTemplateToDocumentVariables(rule.get('annotations').get('subject'))
                    if rule.get('annotations').get('description') is not None:
                        policy.documentation.content = TranslatePromQLTemplateToDocumentVariables(rule.get('annotations').get('description'))
                    policy.documentation.mimeType = 'text/markdown'
                if _VALID_LABEL_REGEXP.fullmatch(group.get('name')) is not None:
                    policy.displayName = '{0}/{1}'.format(group.get('name'), rule.get('alert'))
                else:
                    policy.displayName = '"{0}"/{1}'.format(group.get('name'), rule.get('alert'))
                policy.combiner = arg_utils.ChoiceToEnum('OR', policy.CombinerValueValuesEnum, item_type='combiner')
                if channels is not None:
                    policy.notificationChannels = channels
                policies.append(policy)
        return policies
    except Exception as exc:
        raise YamlOrJsonLoadError('Could not parse YAML: {0}'.format(exc))