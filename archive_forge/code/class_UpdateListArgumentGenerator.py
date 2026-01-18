from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import enum
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import arg_parsers_usage_text as usage_text
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope.concepts import util as format_util
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.apis import yaml_command_schema_util as util
import six
class UpdateListArgumentGenerator(UpdateBasicArgumentGenerator):
    """Update flag generator for list."""

    @property
    def _empty_value(self):
        return []

    @property
    def set_arg(self):
        return self._CreateBasicFlag(flag_type=_ConvertValueType(self), action=self.action, metavar=self.metavar, help_text='Set {} to new value.'.format(self.arg_name))

    @property
    def clear_arg(self):
        return self._CreateBasicFlag(flag_prefix=Prefix.CLEAR, action='store_true', help_text='Clear {} value and set to {}.'.format(self.arg_name, self._GetTextFormatOfEmptyValue(self._empty_value)))

    @property
    def update_arg(self):
        return self._CreateBasicFlag(flag_prefix=Prefix.ADD, flag_type=_ConvertValueType(self), action=self.action, help_text='Add new value to {} list.'.format(self.arg_name))

    @property
    def remove_arg(self):
        return self._CreateBasicFlag(flag_prefix=Prefix.REMOVE, flag_type=_ConvertValueType(self), action=self.action, help_text='Remove existing value from {} list.'.format(self.arg_name))

    def ApplySetFlag(self, existing_val, set_val):
        if set_val:
            return set_val
        return existing_val

    def ApplyClearFlag(self, existing_val, clear_flag):
        if clear_flag:
            return self._empty_value
        return existing_val

    def ApplyRemoveFlag(self, existing_val, remove_val):
        if remove_val:
            return [x for x in existing_val if x not in remove_val]
        return existing_val

    def ApplyUpdateFlag(self, existing_val, update_val):
        if update_val:
            return existing_val + [x for x in update_val if x not in existing_val]
        return existing_val