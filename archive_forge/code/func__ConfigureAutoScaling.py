from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import text
import six
def _ConfigureAutoScaling(self, version, **kwargs):
    """Adds `auto_scaling` fields to version."""
    if not any(kwargs.values()):
        return
    version.autoScaling = self.messages.GoogleCloudMlV1AutoScaling()
    if kwargs['min_nodes']:
        version.autoScaling.minNodes = kwargs['min_nodes']
    if kwargs['max_nodes']:
        version.autoScaling.maxNodes = kwargs['max_nodes']
    if kwargs['metrics']:
        version.autoScaling.metrics = []
        if 'cpu-usage' in kwargs['metrics']:
            t = int(kwargs['metrics']['cpu-usage'])
            version.autoScaling.metrics.append(self.messages.GoogleCloudMlV1MetricSpec(name=self.messages.GoogleCloudMlV1MetricSpec.NameValueValuesEnum.CPU_USAGE, target=t))
        if 'gpu-duty-cycle' in kwargs['metrics']:
            t = int(kwargs['metrics']['gpu-duty-cycle'])
            version.autoScaling.metrics.append(self.messages.GoogleCloudMlV1MetricSpec(name=self.messages.GoogleCloudMlV1MetricSpec.NameValueValuesEnum.GPU_DUTY_CYCLE, target=t))