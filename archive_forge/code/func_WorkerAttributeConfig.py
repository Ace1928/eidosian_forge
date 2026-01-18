from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import abc
import copy
import os
import re
from googlecloudsdk.api_lib.run import global_methods
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.calliope.concepts import util as concepts_util
from googlecloudsdk.command_lib.run import exceptions
from googlecloudsdk.command_lib.run import platforms
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
def WorkerAttributeConfig(prompt=False):
    """Attribute config with fallthrough prompt only if requested."""
    if prompt:
        fallthroughs = [WorkerPromptFallthrough()]
    else:
        fallthroughs = []
    return concepts.ResourceParameterAttributeConfig(name='worker', help_text='Worker for the {resource}.', fallthroughs=fallthroughs)