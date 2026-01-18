from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.active_directory import exceptions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def DomainRemoveRegionFlag():
    """Defines a flag for removing a region."""
    return base.Argument('--remove-region', help='      A region to de-provision this domain from.\n      If domain is already not provisioned in a region, nothing will be done in\n      that region. Domains must be left provisioned in at least one region.\n      Supported regions are: {}.\n      '.format(', '.join(VALID_REGIONS)))