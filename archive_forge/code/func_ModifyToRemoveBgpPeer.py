from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import ipaddress
from apitools.base.py import encoding
from googlecloudsdk.api_lib.edge_cloud.networking import utils
from googlecloudsdk.calliope import parser_errors
from googlecloudsdk.core import exceptions as core_exceptions
import six
def ModifyToRemoveBgpPeer(self, args, existing):
    """Mutate the router to delete BGP peers."""
    input_remove_list = args.peer_names if args.peer_names else []
    input_remove_list = input_remove_list + ([args.peer_name] if args.peer_name else [])
    actual_remove_list = []
    replacement = encoding.CopyProtoMessage(existing)
    existing_router = encoding.CopyProtoMessage(existing)
    for peer in existing_router.bgpPeer:
        if peer.name in input_remove_list:
            replacement.bgpPeer.remove(peer)
            actual_remove_list.append(peer.name)
    not_found_peer = sorted(set(input_remove_list) - set(actual_remove_list))
    if not_found_peer:
        error_msg = 'peer [{}] not found'.format(', '.join(not_found_peer))
        raise core_exceptions.Error(error_msg)
    return replacement