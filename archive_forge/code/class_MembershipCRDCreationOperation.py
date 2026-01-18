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
class MembershipCRDCreationOperation(object):
    """An operation that waits for a membership CRD to be created."""
    CREATED_KEYWORD = 'unchanged'
    CONFIGURED_KEYWORD = 'configured'

    def __init__(self, kube_client, membership_crd_manifest):
        self.kube_client = kube_client
        self.done = False
        self.succeeded = False
        self.error = None
        self.membership_crd_manifest = membership_crd_manifest

    def __str__(self):
        return '<creating membership CRD>'

    def Update(self):
        """Updates this operation with the latest membership creation status."""
        out, err = self.kube_client.CreateMembershipCRD(self.membership_crd_manifest)
        if err:
            self.done = True
            self.error = err
        elif self.CREATED_KEYWORD in out or self.CONFIGURED_KEYWORD in out:
            self.done = True
            self.succeeded = True