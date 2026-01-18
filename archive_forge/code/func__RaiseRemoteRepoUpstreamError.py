from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.artifacts import exceptions as ar_exceptions
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.artifacts import requests as ar_requests
from googlecloudsdk.command_lib.util.apis import arg_utils
def _RaiseRemoteRepoUpstreamError(facade: str, remote_input: str):
    raise ar_exceptions.InvalidInputValueError("Invalid repo upstream for remote repository: '{remote_input}'. Valid choices are: [{enums}].\nIf you intended to enter a custom upstream URI, this value must start with 'https://' or 'http://'.\n".format(remote_input=remote_input, enums=_EnumsStrForFacade(facade)))