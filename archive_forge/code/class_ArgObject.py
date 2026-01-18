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
class ArgObject(arg_utils.ArgObjectType):
    """A wrapper to bind an ArgObject argument to a message or field."""

    @classmethod
    def FromData(cls, data=None):
        """Creates ArgObject from yaml data."""
        spec = data.get('spec')
        return cls(api_field=data['api_field'], arg_name=data.get('arg_name'), help_text=data.get('help_text'), hidden=data.get('hidden'), spec=[ArgObject.FromData(f) for f in spec] if spec is not None else None)

    def __init__(self, api_field=None, arg_name=None, help_text=None, hidden=None, spec=None):
        self.api_field = api_field
        self.arg_name = arg_name
        self.help_text = help_text
        self.hidden = hidden
        self.spec = spec

    def Action(self, field):
        """Returns the correct argument action.

    Args:
      field: apitools field instance

    Returns:
      str, argument action string.
    """
        if field.repeated:
            return arg_parsers.FlattenAction()
        return 'store'

    def _GetFieldTypeFromSpec(self, api_field):
        """Returns first spec field that matches the api_field."""
        default_type = ArgObject()
        spec = self.spec or []
        return next((f for f in spec if f.api_field == api_field), default_type)

    def _GenerateSubFieldType(self, message, api_field, is_label_field=False):
        """Retrieves the the type of the field from messsage.

    Args:
      message: Apitools message class
      api_field: str, field path of message
      is_label_field: bool, whether field is part of labels map field

    Returns:
      _FieldSpecType, Type function that returns apitools message
        instance or list of instances from string value.
    """
        f = arg_utils.GetFieldFromMessage(message, api_field)
        arg_obj = self._GetFieldTypeFromSpec(api_field)
        return arg_obj.GenerateType(f, is_label_field=is_label_field, is_root=False)

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

    def _GenerateMessageType(self, field_spec, is_root=True):
        """Returns function that parses apitools message fields from string.

    Args:
      field_spec: _FieldSpec, information about the field
      is_root: whether the _MessageFieldType is for the root level of
        the message

    Returns:
      _MessageFieldType that takes string like 'foo=bar' or '{"foo": "bar"}' and
      creates an apitools message like Message(foo=bar) or [Message(foo=bar)]
    """
        if self.spec is not None:
            field_names = [f.api_field for f in self.spec]
        else:
            output_only_fields = {'createTime', 'updateTime'}
            field_names = [f.name for f in field_spec.field.type.all_fields() if f.name not in output_only_fields]
        field_specs = [self._GenerateSubFieldType(field_spec.field.type, name) for name in field_names]
        required = [f.arg_name for f in field_specs if f.required]
        arg_obj = arg_parsers.ArgObject(spec={f.arg_name: f for f in field_specs}, help_text=self.help_text, required_keys=required, repeated=field_spec.repeated, hidden=field_spec.hidden, enable_shorthand=is_root)
        return _MessageFieldType(arg_type=arg_obj, field_spec=field_spec, field_specs=field_specs)

    def _GenerateFieldType(self, field_spec, is_label_field=False):
        """Returns _FieldType that parses apitools field from string.

    Args:
      field_spec: _FieldSpec, information about the field
      is_label_field: bool, whether or not the field is for a labels map field.
        If true, supplies default validation and help text.

    Returns:
      _FieldType that takes string like '1' or ['1'] and parses it
      into 1 or [1] depending on the apitools field type
    """
        if is_label_field and field_spec.arg_name == KEY:
            value_type = labels_util.KEY_FORMAT_VALIDATOR
            default_help_text = labels_util.KEY_FORMAT_HELP
        elif is_label_field and field_spec.arg_name == VALUE:
            value_type = labels_util.VALUE_FORMAT_VALIDATOR
            default_help_text = labels_util.VALUE_FORMAT_HELP
        else:
            value_type = _GetFieldValueType(field_spec.field)
            default_help_text = None
        arg_obj = arg_parsers.ArgObject(value_type=value_type, help_text=self.help_text or default_help_text, repeated=field_spec.repeated, hidden=field_spec.hidden, enable_shorthand=False)
        return _FieldType(arg_type=arg_obj, field_spec=field_spec, choices=None)

    def GenerateType(self, field, is_root=True, is_label_field=False):
        """Generates a _FieldSpecType to parse the argument.

    Args:
      field: apitools field instance we are generating ArgObject for
      is_root: bool, whether this is the first level of the ArgObject
        we are generating for.
      is_label_field: bool, whether the field is for labels map field

    Returns:
      _FieldSpecType, Type function that returns apitools message
      instance or list of instances from string value.
    """
        field_spec = _FieldSpec.FromUserData(field, arg_name=self.arg_name, api_field=self.api_field, hidden=self.hidden)
        field_type = arg_utils.GetFieldType(field)
        if field_type == arg_utils.FieldType.MAP:
            return self._GenerateMapType(field_spec, is_root)
        elif field_type == arg_utils.FieldType.MESSAGE:
            return self._GenerateMessageType(field_spec, is_root)
        else:
            return self._GenerateFieldType(field_spec, is_label_field)