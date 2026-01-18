from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.ai import errors
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
import six
def CreateUsingReplicas(config, display_name, base_config, replicas_arg, validate_only, labels=None, etag=None):
    """Create a new instance configs based on provided list of replicas."""
    msgs = apis.GetMessagesModule('spanner', 'v1')
    config_ref = resources.REGISTRY.Parse(base_config, params={'projectsId': properties.VALUES.core.project.GetOrFail}, collection='spanner.projects.instanceConfigs')
    replica_info_list = []
    _AppendReplicas(msgs, replicas_arg, replica_info_list)
    labels_message = {}
    if labels is not None:
        labels_message = msgs.InstanceConfig.LabelsValue(additionalProperties=[msgs.InstanceConfig.LabelsValue.AdditionalProperty(key=key, value=value) for key, value in six.iteritems(labels)])
    return _Create(msgs, config, display_name, config_ref.RelativeName(), replica_info_list, labels_message, validate_only, etag)