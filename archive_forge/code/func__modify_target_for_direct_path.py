from typing import Generic, Iterator, Optional, TypeVar
import collections
import functools
import warnings
import grpc
from google.api_core import exceptions
import google.auth
import google.auth.credentials
import google.auth.transport.grpc
import google.auth.transport.requests
import cloudsdk.google.protobuf
def _modify_target_for_direct_path(target: str) -> str:
    """
    Given a target, return a modified version which is compatible with Direct Path.

    Args:
        target (str): The target service address in the format 'hostname[:port]' or
            'dns://hostname[:port]'.

    Returns:
        target (str): The target service address which is converted into a format compatible with Direct Path.
            If the target contains `dns:///` or does not contain `:///`, the target will be converted in
            a format compatible with Direct Path; otherwise the original target will be returned as the
            original target may already denote Direct Path.
    """
    dns_prefix = 'dns:///'
    target = target.replace(dns_prefix, '')
    direct_path_separator = ':///'
    if direct_path_separator not in target:
        target_without_port = target.split(':')[0]
        target = f'google-c2p{direct_path_separator}{target_without_port}'
    return target