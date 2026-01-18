from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import io
from apitools.base.protorpclite import messages
from apitools.base.py import encoding
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import property_selector
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.compute import exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.backend_services import backend_services_utils
from googlecloudsdk.command_lib.compute.backend_services import flags
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import edit
import six
def GetReferenceNormalizers(self, resource_registry):

    def MakeReferenceNormalizer(field_name, allowed_collections):
        """Returns a function to normalize resource references."""

        def NormalizeReference(reference):
            """Returns normalized URI for field_name."""
            try:
                value_ref = resource_registry.Parse(reference)
            except resources.UnknownCollectionException:
                raise InvalidResourceError('[{field_name}] must be referenced using URIs.'.format(field_name=field_name))
            if value_ref.Collection() not in allowed_collections:
                raise InvalidResourceError('Invalid [{field_name}] reference: [{value}].'.format(field_name=field_name, value=reference))
            return value_ref.SelfLink()
        return NormalizeReference
    return [('healthChecks[]', MakeReferenceNormalizer('healthChecks', ('compute.httpHealthChecks', 'compute.httpsHealthChecks', 'compute.healthChecks', 'compute.regionHealthChecks'))), ('backends[].group', MakeReferenceNormalizer('group', ('compute.instanceGroups', 'compute.regionInstanceGroups')))]