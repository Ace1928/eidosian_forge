from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from apitools.base.py import encoding
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import exceptions as compute_exceptions
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.backend_buckets import (
from googlecloudsdk.command_lib.compute.backend_services import (
from googlecloudsdk.command_lib.compute.url_maps import flags
from googlecloudsdk.command_lib.compute.url_maps import url_maps_utils
from googlecloudsdk.core import properties
import six
def _ModifyBase(client, args, existing):
    """Modifications to the URL map that are shared between release tracks.

  Args:
    client: The compute client.
    args: the argparse arguments that this command was invoked with.
    existing: the existing URL map message.

  Returns:
    A modified URL map message.
  """
    replacement = encoding.CopyProtoMessage(existing)
    if not args.new_hosts and (not args.existing_host):
        new_hosts = ['*']
    else:
        new_hosts = args.new_hosts
    if new_hosts:
        new_hosts = set(new_hosts)
        for host_rule in existing.hostRules:
            for host in host_rule.hosts:
                if host in new_hosts:
                    raise compute_exceptions.ArgumentError('Cannot create a new host rule with host [{0}] because the host is already part of a host rule that references the path matcher [{1}].'.format(host, host_rule.pathMatcher))
        replacement.hostRules.append(client.messages.HostRule(hosts=sorted(new_hosts), pathMatcher=args.path_matcher_name))
    else:
        target_host_rule = None
        for host_rule in existing.hostRules:
            for host in host_rule.hosts:
                if host == args.existing_host:
                    target_host_rule = host_rule
                    break
            if target_host_rule:
                break
        if not target_host_rule:
            raise compute_exceptions.ArgumentError('No host rule with host [{0}] exists. Check your spelling or use [--new-hosts] to create a new host rule.'.format(args.existing_host))
        path_matcher_orphaned = True
        for host_rule in replacement.hostRules:
            if host_rule == target_host_rule:
                host_rule.pathMatcher = args.path_matcher_name
                continue
            if host_rule.pathMatcher == target_host_rule.pathMatcher:
                path_matcher_orphaned = False
                break
        if path_matcher_orphaned:
            if args.delete_orphaned_path_matcher:
                replacement.pathMatchers = [path_matcher for path_matcher in existing.pathMatchers if path_matcher.name != target_host_rule.pathMatcher]
            else:
                raise compute_exceptions.ArgumentError('This operation will orphan the path matcher [{0}]. To delete the orphan path matcher, rerun this command with [--delete-orphaned-path-matcher] or use [gcloud compute url-maps edit] to modify the URL map by hand.'.format(host_rule.pathMatcher))
    return replacement