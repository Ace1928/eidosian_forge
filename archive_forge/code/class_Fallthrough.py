from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import os
from googlecloudsdk.api_lib import apigee
from googlecloudsdk.calliope.concepts import deps
from googlecloudsdk.command_lib.apigee import errors
from googlecloudsdk.core import config
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core.util import files
class Fallthrough(deps.Fallthrough):
    """Base class for Apigee resource argument fallthroughs."""
    _handled_fields = []

    def __init__(self, hint, active=False, plural=False):
        super(Fallthrough, self).__init__(None, hint, active, plural)

    def __contains__(self, field):
        """Returns whether `field` is handled by this fallthrough class."""
        return field in self._handled_fields

    def _Call(self, parsed_args):
        raise NotImplementedError('Subclasses of googlecloudsdk.commnand_lib.apigee.Fallthrough must actually provide a fallthrough.')