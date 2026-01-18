from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def GetClearCertificateMapArgumentForOtherResource(resource_type, required=False):
    """Returns the flag for clearing a Certificate Manager certificate map."""
    return base.Argument('--clear-certificate-map', action='store_true', default=False, required=required, help='Removes any attached certificate map from the {}.'.format(resource_type))