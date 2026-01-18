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
class DeploymentPodsAvailableOperation(object):
    """An operation that tracks whether a Deployment's Pods are all available."""

    def __init__(self, namespace, deployment_name, image, kube_client):
        self.namespace = namespace
        self.deployment_name = deployment_name
        self.image = image
        self.kube_client = kube_client
        self.done = False
        self.succeeded = False
        self.error = None

    def __str__(self):
        return '<Pod availability for {}/{}>'.format(self.namespace, self.deployment_name)

    def Update(self):
        """Updates this operation with the latest Deployment availability status."""
        deployment_resource = 'deployment/{}'.format(self.deployment_name)

        def _HandleErr(err):
            """Updates the operation for the provided error."""
            if 'NotFound' in err:
                return
            self.done = True
            self.succeeded = False
            self.error = err
        deployment_image, err = self.kube_client.GetResourceField(self.namespace, deployment_resource, '.spec.template.spec.containers[0].image')
        if err:
            _HandleErr(err)
            return
        if deployment_image != self.image:
            return
        spec_replicas, err = self.kube_client.GetResourceField(self.namespace, deployment_resource, '.spec.replicas')
        if err:
            _HandleErr(err)
            return
        status_replicas, err = self.kube_client.GetResourceField(self.namespace, deployment_resource, '.status.replicas')
        if err:
            _HandleErr(err)
            return
        available_replicas, err = self.kube_client.GetResourceField(self.namespace, deployment_resource, '.status.availableReplicas')
        if err:
            _HandleErr(err)
            return
        updated_replicas, err = self.kube_client.GetResourceField(self.namespace, deployment_resource, '.status.updatedReplicas')
        if err:
            _HandleErr(err)
            return
        if updated_replicas < spec_replicas:
            return
        if status_replicas > updated_replicas:
            return
        if available_replicas < updated_replicas:
            return
        self.succeeded = True
        self.done = True