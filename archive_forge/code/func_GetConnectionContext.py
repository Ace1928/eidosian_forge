from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import abc
import base64
import contextlib
import os
import re
import ssl
import sys
import tempfile
from googlecloudsdk.api_lib.run import gke
from googlecloudsdk.api_lib.run import global_methods
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.run import exceptions as serverless_exceptions
from googlecloudsdk.command_lib.run import flags
from googlecloudsdk.command_lib.run import platforms
from googlecloudsdk.core import properties
from googlecloudsdk.core import transport
from googlecloudsdk.core.util import files
import requests
import six
from six.moves.urllib import parse as urlparse
def GetConnectionContext(args, product=flags.Product.RUN, release_track=base.ReleaseTrack.GA, version_override=None, platform=None, region_label=None, is_multiregion=False):
    """Gets the regional, kubeconfig, or GKE connection context.

  Args:
    args: Namespace, the args namespace.
    product: Which product is requesting connection context.
    release_track: Release track of the command being run.
    version_override: If specified, the given api version will be used no matter
      the other parameters.
    platform: 'gke', 'kubernetes', or 'managed'. If not specified, the value of
      the --platform flag will be used instead.
    region_label: A k8s label representing the intended region.
    is_multiregion: Whether we will use the managed Multi-region API.

  Raises:
    ArgumentError if region or cluster is not specified.

  Returns:
    A GKE or regional ConnectionInfo object.
  """
    if platform is None:
        platform = platforms.GetPlatform()
    if platform == platforms.PLATFORM_KUBERNETES:
        kubeconfig = flags.GetKubeconfig(getattr(args, 'kubeconfig', None))
        api_name = _GetApiName(product, release_track, is_cluster=True)
        api_version = _GetApiVersion(product, release_track, is_cluster=True, version_override=version_override)
        return KubeconfigConnectionContext(kubeconfig, api_name, api_version, args.context)
    if platform == platforms.PLATFORM_GKE:
        cluster_ref = args.CONCEPTS.cluster.Parse()
        if not cluster_ref:
            raise serverless_exceptions.ArgumentError('You must specify a cluster in a given location. Either use the `--cluster` and `--cluster-location` flags or set the run/cluster and run/cluster_location properties.')
        api_name = _GetApiName(product, release_track, is_cluster=True)
        api_version = _GetApiVersion(product, release_track, is_cluster=True, version_override=version_override)
        return GKEConnectionContext(cluster_ref, api_name, api_version)
    if platform == platforms.PLATFORM_MANAGED:
        api_name = _GetApiName(product, release_track)
        api_version = _GetApiVersion(product, release_track, version_override=version_override)
        if not is_multiregion:
            region = flags.GetRegion(args, prompt=True, region_label=region_label)
            if not region:
                raise serverless_exceptions.ArgumentError('You must specify a region. Either use the `--region` flag or set the run/region property.')
            return RegionalConnectionContext(region, api_name, api_version)
        else:
            region_list = flags.GetMultiRegion(args)
            return MultiRegionConnectionContext(api_name, api_version, region_list)