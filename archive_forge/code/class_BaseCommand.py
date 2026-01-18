from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import argparse  # pylint: disable=unused-import
import json
import textwrap
from apitools.base.py import base_api  # pylint: disable=unused-import
import enum
from googlecloudsdk.api_lib.compute import base_classes_resource_registry as resource_registry
from googlecloudsdk.api_lib.compute import client_adapter
from googlecloudsdk.api_lib.compute import constants
from googlecloudsdk.api_lib.compute import lister
from googlecloudsdk.api_lib.compute import property_selector
from googlecloudsdk.api_lib.compute import request_helper
from googlecloudsdk.api_lib.compute import resource_specs
from googlecloudsdk.api_lib.compute import scope_prompter
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute import completers
from googlecloudsdk.command_lib.compute import exceptions as compute_exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import text
import six
class BaseCommand(base.Command, scope_prompter.ScopePrompter):
    """Base class for all compute subcommands."""

    def __init__(self, *args, **kwargs):
        super(BaseCommand, self).__init__(*args, **kwargs)
        self.__resource_spec = None
        self._project = properties.VALUES.core.project.Get(required=True)
        self._compute_holder = ComputeApiHolder(self.ReleaseTrack())

    @property
    def _resource_spec(self):
        if not self.resource_type:
            return None
        if self.__resource_spec is None:
            self.__resource_spec = resource_specs.GetSpec(self.resource_type, self.messages, self.compute_client.api_version)
        return self.__resource_spec

    @property
    def transformations(self):
        if self._resource_spec:
            return self._resource_spec.transformations
        else:
            return None

    @property
    def resource_type(self):
        """Specifies the name of the collection that should be printed."""
        return None

    @property
    def http(self):
        """Specifies the http client to be used for requests."""
        return self.compute_client.apitools_client.http

    @property
    def project(self):
        """Specifies the user's project."""
        return self._project

    @property
    def batch_url(self):
        """Specifies the API batch URL."""
        return self.compute_client.batch_url

    @property
    def compute_client(self):
        """Specifies the compute client."""
        return self._compute_holder.client

    @property
    def compute(self):
        """Specifies the compute client."""
        return self.compute_client.apitools_client

    @property
    def resources(self):
        """Specifies the resources parser for compute resources."""
        return self._compute_holder.resources

    @property
    def messages(self):
        """Specifies the API message classes."""
        return self.compute_client.messages

    def Collection(self):
        """Returns the resource collection path."""
        return 'compute.' + self.resource_type if self.resource_type else None