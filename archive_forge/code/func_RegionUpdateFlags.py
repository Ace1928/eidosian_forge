from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.active_directory import exceptions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def RegionUpdateFlags():
    """Defines flags for updating regions."""
    region_group = base.ArgumentGroup(mutex=True)
    region_group.AddArgument(DomainAddRegionFlag())
    region_group.AddArgument(DomainRemoveRegionFlag())
    return region_group