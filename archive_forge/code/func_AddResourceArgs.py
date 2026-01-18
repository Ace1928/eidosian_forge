from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.command_lib.compute.images.packages import resource_args
from googlecloudsdk.command_lib.util.concepts import concept_parsers
def AddResourceArgs(parser):
    """Add resource args of images for 'packages diff' command."""
    concept_parsers.ConceptParser([resource_args.CreateImageResourcePresentationSpec('Name of the disk image as the diff base.', 'base'), resource_args.CreateImageResourcePresentationSpec('Name of the disk image to diff with base image.', 'diff')]).AddToParser(parser)