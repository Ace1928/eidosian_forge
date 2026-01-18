from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import json
import re
from googlecloudsdk.api_lib.apigee import base
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.command_lib.apigee import errors
from googlecloudsdk.command_lib.apigee import request
from googlecloudsdk.command_lib.apigee import resource_args
from googlecloudsdk.core import log
@classmethod
def CreateArchiveDeployment(cls, identifiers, post_data):
    """Apigee API for creating a new archive deployment.

    Args:
      identifiers: A dict of identifiers for the request entity path, which must
        include "organizationsId" and "environmentsId".
      post_data: A dict of the request body to include in the
        CreateArchiveDeployment API call.

    Returns:
      A dict of the API response. The API call starts a long-running operation,
        so the response dict will contain info about the operation id.

    Raises:
      command_lib.apigee.errors.RequestError if there is an error with the API
        request.
    """
    try:
        return request.ResponseToApiRequest(identifiers, cls._entity_path[:-1], cls._entity_path[-1], method='POST', body=json.dumps(post_data))
    except errors.RequestError as error:
        raise error.RewrittenError('archive deployment', 'create')