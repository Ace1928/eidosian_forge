from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
from googlecloudsdk.api_lib.orgpolicy import utils
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.generated_clients.apis.orgpolicy.v2 import orgpolicy_v2_messages
def BuildEmptyPolicy(self, name, has_spec=False, has_dry_run_spec=False):
    spec = None
    dry_run_spec = None
    if has_spec:
        spec = self.messages.GoogleCloudOrgpolicyV2PolicySpec()
    if has_dry_run_spec:
        dry_run_spec = self.messages.GoogleCloudOrgpolicyV2PolicySpec()
    return self.messages.GoogleCloudOrgpolicyV2Policy(name=name, spec=spec, dryRunSpec=dry_run_spec)