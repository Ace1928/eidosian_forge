from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import io
import os
import re
import stat
import textwrap
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import lister
from googlecloudsdk.api_lib.compute import path_simplifier
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import ssh_utils
from googlecloudsdk.command_lib.util.ssh import ssh
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
from googlecloudsdk.core.util import platforms
def _CreateAlias(instance_resource):
    """Returns the alias for the given instance."""
    parts = [instance_resource.name, path_simplifier.Name(instance_resource.zone), properties.VALUES.core.project.Get(required=True)]
    return '.'.join(parts)