from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
from apitools.base.py import encoding
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.app import appengine_api_client
from googlecloudsdk.api_lib.app import operations_util
from googlecloudsdk.api_lib.cloudresourcemanager import projects_api
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.iam import iam_util
from googlecloudsdk.command_lib.projects import util as projects_util
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
import six
def _SetIamPolicy(self, resource_ref, policy):
    policy.version = iam_util.MAX_LIBRARY_IAM_SUPPORTED_VERSION
    request = self.messages.IapSetIamPolicyRequest(resource=resource_ref.RelativeName(), setIamPolicyRequest=self.messages.SetIamPolicyRequest(policy=policy))
    response = self.service.SetIamPolicy(request)
    iam_util.LogSetIamPolicy(resource_ref.RelativeName(), self._Name())
    return response