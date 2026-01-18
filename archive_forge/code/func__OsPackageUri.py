from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.artifacts import exceptions as ar_exceptions
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.artifacts import requests as ar_requests
from googlecloudsdk.command_lib.util.apis import arg_utils
def _OsPackageUri(remote_base, remote_path):
    if not remote_path:
        return remote_base
    if remote_base[-1] != '/':
        remote_base = remote_base + '/'
    return remote_base + remote_path