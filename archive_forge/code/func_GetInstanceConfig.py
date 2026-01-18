from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
import re
from apitools.base.py import list_pager
from cloudsdk.google.protobuf import timestamp_pb2
from googlecloudsdk.api_lib.spanner import response_util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.command_lib.iam import iam_util
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
def GetInstanceConfig(instance):
    """Get the instance config of the passed instance."""
    client = apis.GetClientInstance(_SPANNER_API_NAME, _SPANNER_API_VERSION)
    msgs = apis.GetMessagesModule(_SPANNER_API_NAME, _SPANNER_API_VERSION)
    instance_ref = resources.REGISTRY.Parse(instance, params={'projectsId': properties.VALUES.core.project.GetOrFail}, collection='spanner.projects.instances')
    instance_req = msgs.SpannerProjectsInstancesGetRequest(name=instance_ref.RelativeName(), fieldMask='config')
    instance_info = client.projects_instances.Get(instance_req)
    instance_config = re.search('.*/instanceConfigs/(.*)', instance_info.config).group(1)
    return instance_config