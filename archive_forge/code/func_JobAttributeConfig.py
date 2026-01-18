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
def JobAttributeConfig(prompt=False):
    if prompt:
        fallthroughs = [JobPromptFallthrough()]
    else:
        fallthroughs = []
    return concepts.ResourceParameterAttributeConfig(name='jobs', help_text='Job for the {resource}.', fallthroughs=fallthroughs)