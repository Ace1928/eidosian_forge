from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import datetime
import os
import re
import sys
import time
from apitools.base.py import list_pager
from apitools.base.py.exceptions import HttpNotFoundError
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute.operations import poller
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute import ssh_utils
from googlecloudsdk.command_lib.compute.instances import flags as instance_flags
from googlecloudsdk.command_lib.projects import util as p_util
from googlecloudsdk.command_lib.util.ssh import ssh
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.util import retry
from googlecloudsdk.core.util import times
import six
def AddTpuUserAgent(self, tpu_user_agent):
    """AddTPUUserAgent adds the TPU user agent to enable Cloud Storage access and send logging."""
    project = properties.VALUES.core.project.Get(required=True)
    get_iam_policy_request = self.messages.CloudresourcemanagerProjectsGetIamPolicyRequest(resource=project)
    policy = self.client.projects.GetIamPolicy(get_iam_policy_request)
    policy = self._AddAgentToPolicy(policy, tpu_user_agent)
    if policy is None:
        log.status.Print('TPU Service account:{} has already been enabled'.format(tpu_user_agent))
    else:
        set_iam_policy_request = self.messages.CloudresourcemanagerProjectsSetIamPolicyRequest(resource=project, setIamPolicyRequest=self.messages.SetIamPolicyRequest(policy=policy))
        self.client.projects.SetIamPolicy(set_iam_policy_request)
        log.status.Print('Added Storage and Logging permissions to TPU Service Account:{}'.format(tpu_user_agent))