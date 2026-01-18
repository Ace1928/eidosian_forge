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
class ComputeApiHolder(object):
    """Convenience class to hold lazy initialized client and resources."""

    def __init__(self, release_track, no_http=False):
        if release_track == base.ReleaseTrack.ALPHA:
            self._api_version = 'alpha'
        elif release_track == base.ReleaseTrack.BETA:
            self._api_version = 'beta'
        else:
            self._api_version = 'v1'
        self._client = None
        self._resources = None
        self._no_http = no_http

    @property
    def client(self):
        """Specifies the compute client."""
        if self._client is None:
            self._client = client_adapter.ClientAdapter(self._api_version, self._no_http)
        return self._client

    @property
    def resources(self):
        """Specifies the resources parser for compute resources."""
        if self._resources is None:
            self._resources = resources.REGISTRY.Clone()
            self._resources.RegisterApiByName('compute', self._api_version)
        return self._resources