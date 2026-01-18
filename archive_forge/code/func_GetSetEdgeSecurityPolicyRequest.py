from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import cdn_flags_utils as cdn_flags
from googlecloudsdk.command_lib.compute import exceptions
from googlecloudsdk.command_lib.compute import signed_url_flags
from googlecloudsdk.command_lib.compute.backend_buckets import backend_buckets_utils
from googlecloudsdk.command_lib.compute.backend_buckets import flags as backend_buckets_flags
from googlecloudsdk.command_lib.compute.security_policies import (
from googlecloudsdk.core import log
def GetSetEdgeSecurityPolicyRequest(self, client, backend_bucket_ref, security_policy_ref):
    """Returns a request to set the edge policy for the backend bucket."""
    return (client.apitools_client.backendBuckets, 'SetEdgeSecurityPolicy', client.messages.ComputeBackendBucketsSetEdgeSecurityPolicyRequest(project=backend_bucket_ref.project, backendBucket=backend_bucket_ref.Name(), securityPolicyReference=client.messages.SecurityPolicyReference(securityPolicy=security_policy_ref)))