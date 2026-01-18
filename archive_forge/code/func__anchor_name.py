from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import itertools
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.calliope.concepts import multitype
from googlecloudsdk.calliope.concepts import util as resource_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.apis import registry
from googlecloudsdk.command_lib.util.apis import update_args
from googlecloudsdk.command_lib.util.apis import update_resource_args
from googlecloudsdk.command_lib.util.apis import yaml_command_schema_util as util
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
from googlecloudsdk.core.util import text
@property
def _anchor_name(self):
    """Name of the anchor attribute.

    For anchor attribute foo-bar, the expected format is...
      1. `foo-bar` if anchor is not positional
      2. `FOO_BAR` if anchor is positional
    """
    if self.flag_name_override:
        return self.flag_name_override
    else:
        count = 2 if self.repeated else 1
        return text.Pluralize(count, self._resource_spec.anchor.name)