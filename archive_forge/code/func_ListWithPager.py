from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.iam import iam_util
def ListWithPager(self, secret_ref, limit, request_filter=None, secret_location=None):
    """List secrets returning a pager object."""
    request = self.messages.SecretmanagerProjectsSecretsVersionsListRequest(parent=GetRelativeName(secret_ref, secret_location), filter=request_filter, pageSize=0)
    return list_pager.YieldFromList(service=self.service, request=request, field='versions', limit=limit, batch_size=0, batch_size_attribute='pageSize')