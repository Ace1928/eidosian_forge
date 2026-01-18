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
class UpdateMapArgumentGenerator(UpdateBasicArgumentGenerator):
    """Update flag generator for key-value pairs ie proto map fields."""

    @property
    def _empty_value(self):
        return {}

    @property
    def _is_list_field(self):
        return self.field.name == arg_utils.ADDITIONAL_PROPS

    def _WrapOutput(self, output_list):
        """Wraps field AdditionalProperties in apitools message if needed.

    Args:
      output_list: list of apitools AdditionalProperties messages.

    Returns:
      apitools message instance.
    """
        if self._is_list_field:
            return output_list
        message = self.field.type()
        arg_utils.SetFieldInMessage(message, arg_utils.ADDITIONAL_PROPS, output_list)
        return message

    def _GetPropsFieldValue(self, field):
        """Retrieves AdditionalProperties field value.

    Args:
      field: apitools instance that contains AdditionalProperties field

    Returns:
      list of apitools AdditionalProperties messages.
    """
        if not field:
            return []
        if self._is_list_field:
            return field
        return arg_utils.GetFieldValueFromMessage(field, arg_utils.ADDITIONAL_PROPS)

    @property
    def set_arg(self):
        return self._CreateBasicFlag(flag_type=_ConvertValueType(self), action=self.action, metavar=self.metavar, help_text='Set {} to new value.'.format(self.arg_name))

    @property
    def clear_arg(self):
        return self._CreateBasicFlag(flag_prefix=Prefix.CLEAR, action='store_true', help_text='Clear {} value and set to {}.'.format(self.arg_name, self._GetTextFormatOfEmptyValue(self._empty_value)))

    @property
    def update_arg(self):
        return self._CreateBasicFlag(flag_prefix=Prefix.UPDATE, flag_type=_ConvertValueType(self), action=self.action, help_text='Update {} value or add key value pair.'.format(self.arg_name))

    @property
    def remove_arg(self):
        if self._is_list_field:
            field = self.field
        else:
            field = arg_utils.GetFieldFromMessage(self.field.type, arg_utils.ADDITIONAL_PROPS)
        key_field = arg_utils.GetFieldFromMessage(field.type, 'key')
        key_type = key_field.type or arg_utils.TYPES.get(key_field.variant)
        key_list = arg_parsers.ArgList(element_type=key_type)
        return self._CreateBasicFlag(flag_prefix=Prefix.REMOVE, flag_type=key_list, action='store', help_text='Remove existing value from map {}.'.format(self.arg_name))

    def ApplySetFlag(self, existing_val, set_val):
        if set_val:
            return set_val
        return existing_val

    def ApplyClearFlag(self, existing_val, clear_flag):
        if clear_flag:
            return self._WrapOutput([])
        return existing_val

    def ApplyUpdateFlag(self, existing_val, update_val):
        if update_val:
            output_list = self._GetPropsFieldValue(existing_val)
            update_val_list = self._GetPropsFieldValue(update_val)
            update_key_set = set([x.key for x in update_val_list])
            deduped_list = [x for x in output_list if x.key not in update_key_set]
            return self._WrapOutput(deduped_list + update_val_list)
        return existing_val

    def ApplyRemoveFlag(self, existing_val, remove_val):
        if remove_val:
            output_list = self._GetPropsFieldValue(existing_val)
            remove_val_set = set(remove_val)
            return self._WrapOutput([x for x in output_list if x.key not in remove_val_set])
        return existing_val