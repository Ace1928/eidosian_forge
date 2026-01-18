from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.apphub import consts as api_lib_consts
from googlecloudsdk.api_lib.apphub import utils as api_lib_utils
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.iam import iam_util
def _SetIamPolicyHelper(self, app_id, policy):
    set_req = self.messages.ApphubProjectsLocationsApplicationsSetIamPolicyRequest(resource=app_id, setIamPolicyRequest=self.messages.SetIamPolicyRequest(policy=policy))
    return self._app_client.SetIamPolicy(set_req)