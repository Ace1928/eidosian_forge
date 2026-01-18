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
def TranslatePromQLTemplateToDocumentVariables(template):
    """Translate Prometheus templating language constructs to document variables.

  TranslatePromQLTemplateToDocumentVariables translates common Prometheus
  templating language constructs to their equivalent Cloud Alerting document
  variables. See:
  https://prometheus.io/docs/prometheus/latest/configuration/template_reference/
  and https://cloud.google.com/monitoring/alerts/doc-variables.

  Only the following constructs will be translated:

  "{{ $value }}" will be translated to "${metric.label.value}".
  "{{ humanize $value }}" will be translated to "${metric.label.value}".
  "{{ $labels.<name> }}" will be translated to
  "${metric_or_resource.label.<name>}".
  "{{ humanize $labels.<name> }}" will be translated to
  "${metric_or_resource.label.<name>}".
  "{{ $labels }}" will be translated to
  "${metric_or_resource.labels}".
  "{{ humanize $labels }}" will be translated to
  "${metric_or_resource.labels}".

  The number of spaces inside the curly braces is immaterial.

  All other Prometheus templating language constructs are not translated.

  Notes:
  1. A document variable reference that does not match a variable
     will be rendered as "(none)".
  2. We do not know whether a {{ $labels.<name> }} construct refers to
     a Cloud Alerting metric or a resource label. Thus we translate it to
     "${metric_or_resource.label.<name>}".
     Note that a reference to a non-existent label will be rendered as "(none)".

  Examples:
  1. "[{{$labels.a}}] VALUE = {{ $value }}" will be translated to
     "[${metric_or_resource.label.a}] VALUE = ${metric.label.value}".

  2. "[{{humanize $labels.a}}] VALUE = {{ humanize $value }}"
     will be translated to
     "[${metric_or_resource.label.a}] VALUE = ${metric.label.value}".

  Args:
    template: String contents of the "subject" or "content" fields of an
    AlertPolicy protoco buffer. The contents of these fields is a template
    which may contain Prometheus templating language constructs.

  Returns:
    The translated template.
  """
    return _LABELS_KEY_REGEXP.sub('${metric_or_resource.label.\\2}', _LABELS_VARIABLE_REGEXP.sub('${metric_or_resource.labels}', _VALUE_VARIABLE_REGEXP.sub('${metric.label.value}', template)))