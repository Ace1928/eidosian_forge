from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import io
import json
import os
import re
from googlecloudsdk.api_lib.container import api_adapter as gke_api_adapter
from googlecloudsdk.api_lib.container import kubeconfig as kconfig
from googlecloudsdk.api_lib.container import util as c_util
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.container.fleet import format_util
from googlecloudsdk.command_lib.container.fleet.memberships import gke_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import requests
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
from kubernetes import client as kube_client_lib
from kubernetes import config as kube_client_config
from six.moves.urllib.parse import urljoin
def ApplyMembership(self, membership_crd_manifest, membership_cr_manifest):
    """Apply membership resources."""
    if membership_crd_manifest:
        _, error = waiter.WaitFor(KubernetesPoller(), MembershipCRDCreationOperation(self, membership_crd_manifest), pre_start_sleep_ms=NAMESPACE_DELETION_INITIAL_WAIT_MS, max_wait_ms=NAMESPACE_DELETION_TIMEOUT_MS, wait_ceiling_ms=NAMESPACE_DELETION_MAX_POLL_INTERVAL_MS, sleep_ms=NAMESPACE_DELETION_INITIAL_POLL_INTERVAL_MS)
        if error:
            raise exceptions.Error('Membership CRD creation failed to complete: {}'.format(error))
    if membership_cr_manifest:
        _, err = self.Apply(membership_cr_manifest)
        if err:
            raise exceptions.Error('Failed to apply Membership CR to cluster: {}'.format(err))