from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import io
from apitools.base.protorpclite import messages
from apitools.base.py import encoding
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import property_selector
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import exceptions as compute_exceptions
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.url_maps import flags
from googlecloudsdk.command_lib.compute.url_maps import url_maps_utils
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import edit
import six
def _GetReferenceNormalizers(resource_registry):
    """Gets normalizers that translate short names to URIs."""

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
    allowed_collections = ['compute.backendServices', 'compute.backendBuckets', 'compute.regionBackendServices']
    return [('defaultService', MakeReferenceNormalizer('defaultService', allowed_collections)), ('pathMatchers[].defaultService', MakeReferenceNormalizer('defaultService', allowed_collections)), ('pathMatchers[].pathRules[].service', MakeReferenceNormalizer('service', allowed_collections)), ('tests[].service', MakeReferenceNormalizer('service', allowed_collections))]