from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import routers_utils
from googlecloudsdk.calliope import parser_errors
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core.console import console_io
import six
def GenerateMd5AuthenticationKeyName(router_message, args):
    """Generates an MD5 authentication key name for the BGP peer.

  Args:
    router_message: the Cloud Router that contains the relevant BGP peer.
    args: contains arguments passed to the command

  Returns:
    Generated MD5 authentication key name
  """
    md5_authentication_key_names = set()
    for bgp_peer in router_message.bgpPeers:
        if bgp_peer.md5AuthenticationKeyName is not None:
            md5_authentication_key_names.add(bgp_peer.md5AuthenticationKeyName)
    substrings_max_length = _MAX_LENGTH_OF_MD5_AUTHENTICATION_KEY - len(_MD5_AUTHENTICATION_KEY_SUFFIX)
    md5_authentication_key_name = args.peer_name[:substrings_max_length] + _MD5_AUTHENTICATION_KEY_SUFFIX
    md5_authentication_key_name_suffix = 2
    while md5_authentication_key_name in md5_authentication_key_names:
        substrings_max_length = _MAX_LENGTH_OF_MD5_AUTHENTICATION_KEY - len(_MD5_AUTHENTICATION_KEY_SUBSTRING) - len(six.text_type(md5_authentication_key_name_suffix))
        md5_authentication_key_name = args.peer_name[:substrings_max_length] + _MD5_AUTHENTICATION_KEY_SUBSTRING + six.text_type(md5_authentication_key_name_suffix)
        md5_authentication_key_name_suffix += 1
    return md5_authentication_key_name