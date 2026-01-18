from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import time
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import http_wrapper
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.container import constants as gke_constants
from googlecloudsdk.api_lib.container import util
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources as cloud_resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.console import progress_tracker
from googlecloudsdk.core.util import times
import six
from six.moves import range  # pylint: disable=redefined-builtin
import six.moves.http_client
from cmd argument to set a surge upgrade strategy.
def ModifyBinaryAuthorization(self, cluster_ref, existing_binauthz_config, enable_binauthz, binauthz_evaluation_mode, binauthz_policy_bindings):
    """Updates the binary_authorization message."""
    if existing_binauthz_config is not None:
        binary_authorization = self.messages.BinaryAuthorization(evaluationMode=existing_binauthz_config.evaluationMode, policyBindings=existing_binauthz_config.policyBindings)
    else:
        binary_authorization = self.messages.BinaryAuthorization()
    if enable_binauthz is not None:
        if enable_binauthz and BinauthzEvaluationModeRequiresPolicy(self.messages, binary_authorization.evaluationMode):
            console_io.PromptContinue(message='This will cause the current version of Binary Authorization to be downgraded (not recommended).', cancel_on_no=True)
        binary_authorization = self.messages.BinaryAuthorization(enabled=enable_binauthz)
    else:
        if binauthz_evaluation_mode is not None:
            binary_authorization.evaluationMode = util.GetBinauthzEvaluationModeMapper(self.messages, hidden=False).GetEnumForChoice(binauthz_evaluation_mode)
            if not BinauthzEvaluationModeRequiresPolicy(self.messages, binary_authorization.evaluationMode):
                binary_authorization.policyBindings = []
        if binauthz_policy_bindings is not None:
            binary_authorization.policyBindings = []
            for binding in binauthz_policy_bindings:
                binary_authorization.policyBindings.append(self.messages.PolicyBinding(name=binding['name']))
    update = self.messages.ClusterUpdate(desiredBinaryAuthorization=binary_authorization)
    op = self.client.projects_locations_clusters.Update(self.messages.UpdateClusterRequest(name=ProjectLocationCluster(cluster_ref.projectId, cluster_ref.zone, cluster_ref.clusterId), update=update))
    return self.ParseOperation(op.name, cluster_ref.zone)