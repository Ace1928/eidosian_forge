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
class APIsClient(base.BaseClient):
    _entity_path = ['organization', 'api']

    @classmethod
    def Deploy(cls, identifiers, override=False):
        deployment_path = ['organization', 'environment', 'api', 'revision']
        query_params = {'override': 'true'} if override else {}
        try:
            return request.ResponseToApiRequest(identifiers, deployment_path, 'deployment', method='POST', query_params=query_params)
        except errors.RequestError as error:
            raise error.RewrittenError('API proxy', 'deploy')

    @classmethod
    def Undeploy(cls, identifiers):
        try:
            return request.ResponseToApiRequest(identifiers, ['organization', 'environment', 'api', 'revision'], 'deployment', method='DELETE')
        except errors.RequestError as error:
            raise error.RewrittenError('deployment', 'undeploy')