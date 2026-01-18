from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.protorpclite import messages
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.apis import update
from googlecloudsdk.command_lib.util.apis import yaml_arg_schema
from googlecloudsdk.command_lib.util.apis import yaml_command_schema
from googlecloudsdk.command_lib.util.apis import yaml_command_schema_util as util
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import resources
from googlecloudsdk.core.resource import resource_property
def _AddLabelsToUpdateMask(static_field, update_mask_path):
    if update_mask_path not in static_field or not static_field[update_mask_path]:
        static_field[update_mask_path] = 'labels'
        return
    if 'labels' in static_field[update_mask_path].split(','):
        return
    static_field[update_mask_path] = static_field[update_mask_path] + ',' + 'labels'