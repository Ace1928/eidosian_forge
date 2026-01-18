from __future__ import absolute_import
from __future__ import annotations
from __future__ import division
from __future__ import unicode_literals
import abc
from collections.abc import Callable
import dataclasses
from typing import Any
from apitools.base.protorpclite import messages as apitools_messages
from googlecloudsdk.calliope import actions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import arg_parsers_usage_text as usage_text
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import module_util
def _GenerateMapType(self, field_spec, is_root=True):
    """Returns function that parses apitools map fields from string.

    Map fields are proto fields with type `map<...>` that generate
    apitools message with an additionalProperties field

    Args:
      field_spec: _FieldSpec, information about the field
      is_root: whether the type function is for the root level of the message

    Returns:
      type function that takes string like 'foo=bar' or '{"foo": "bar"}' and
        creates an apitools message additionalProperties field
    """
    try:
        additional_props_field = arg_utils.GetFieldFromMessage(field_spec.field.type, arg_utils.ADDITIONAL_PROPS)
    except arg_utils.UnknownFieldError:
        raise InvalidSchemaError('{name} message does not contain field "{props}". Remove "{props}" from api field name.'.format(name=field_spec.api_field, props=arg_utils.ADDITIONAL_PROPS))
    is_label_field = field_spec.arg_name == 'labels'
    props_field_spec = _FieldSpec.FromUserData(additional_props_field, arg_name=self.arg_name)
    key_type = self._GenerateSubFieldType(additional_props_field.type, KEY, is_label_field=is_label_field)
    value_type = self._GenerateSubFieldType(additional_props_field.type, VALUE, is_label_field=is_label_field)
    arg_obj = arg_parsers.ArgObject(key_type=key_type, value_type=value_type, help_text=self.help_text, hidden=field_spec.hidden, enable_shorthand=is_root)
    additional_prop_spec_type = _AdditionalPropsType(arg_type=arg_obj, field_spec=props_field_spec, key_spec=key_type, value_spec=value_type)
    return _MapFieldType(arg_type=additional_prop_spec_type, field_spec=field_spec)