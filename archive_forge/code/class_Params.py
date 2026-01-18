from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from frozendict import frozendict
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.run.integrations import flags
from googlecloudsdk.command_lib.run.integrations import run_apps_operations
from googlecloudsdk.command_lib.run.integrations import types_describe_printer
from googlecloudsdk.command_lib.runapps import exceptions
from googlecloudsdk.core.resource import resource_printer
class Params:
    """Simple struct like class that only holds data."""

    def __init__(self, required, optional):
        self.required = required
        self.optional = optional