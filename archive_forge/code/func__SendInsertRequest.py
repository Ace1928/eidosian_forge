from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import exceptions as compute_exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute import operation_utils
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.target_http_proxies import flags
from googlecloudsdk.command_lib.compute.target_http_proxies import target_http_proxies_utils
from googlecloudsdk.command_lib.export import util as export_util
from googlecloudsdk.core import yaml_validator
from googlecloudsdk.core.console import console_io
def _SendInsertRequest(client, resources, target_http_proxy_ref, target_http_proxy):
    """Sends Target HTTP Proxy insert request."""
    if target_http_proxies_utils.IsRegionalTargetHttpProxiesRef(target_http_proxy_ref):
        service = client.apitools_client.regionTargetHttpProxies
        operation = service.Insert(client.messages.ComputeRegionTargetHttpProxiesInsertRequest(project=target_http_proxy_ref.project, region=target_http_proxy_ref.region, targetHttpProxy=target_http_proxy))
    else:
        service = client.apitools_client.targetHttpProxies
        operation = service.Insert(client.messages.ComputeTargetHttpProxiesInsertRequest(project=target_http_proxy_ref.project, targetHttpProxy=target_http_proxy))
    return _WaitForOperation(resources, service, operation, target_http_proxy_ref, 'Inserting TargetHttpProxy')