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
class UpdateBasicArgumentGenerator(UpdateArgumentGenerator):
    """Update flag generator for simple flags."""

    @classmethod
    def FromArgData(cls, arg_data, field):
        """Creates a flag generator from yaml arg data and request message.

    Args:
      arg_data: yaml_arg_schema.Argument, data about flag being generated
      field: messages.Field, apitools field instance.

    Returns:
      UpdateArgumentGenerator, the correct version of flag generator
    """
        flag_type, action = arg_utils.GenerateFlagType(field, arg_data)
        is_repeated = field.repeated if arg_data.repeated is None else arg_data.repeated
        field_type = arg_utils.GetFieldType(field)
        if field_type == arg_utils.FieldType.MAP:
            gen_cls = UpdateMapArgumentGenerator
        elif is_repeated:
            gen_cls = UpdateListArgumentGenerator
        else:
            gen_cls = UpdateDefaultArgumentGenerator
        return gen_cls(arg_name=arg_data.arg_name, flag_type=flag_type, field=field, action=action, is_hidden=arg_data.hidden, help_text=arg_data.help_text, api_field=arg_data.api_field, repeated=arg_data.repeated, processor=arg_data.processor, choices=arg_data.choices, metavar=arg_data.metavar)

    def __init__(self, arg_name, flag_type=None, field=None, action=None, is_hidden=False, help_text=None, api_field=None, repeated=False, processor=None, choices=None, metavar=None):
        super(UpdateBasicArgumentGenerator, self).__init__()
        self.arg_name = format_util.NormalizeFormat(arg_name)
        self.field = field
        self.flag_type = flag_type
        self.action = action
        self.is_hidden = is_hidden
        self.help_text = help_text
        self.api_field = api_field
        self.repeated = repeated
        self.processor = processor
        self.choices = choices
        self.metavar = metavar

    def GetArgFromNamespace(self, namespace, arg):
        if arg is None:
            return None
        return arg_utils.GetFromNamespace(namespace, arg.name)

    def GetFieldValueFromMessage(self, existing_message):
        """Retrieves existing field from message."""
        if existing_message:
            existing_value = arg_utils.GetFieldValueFromMessage(existing_message, self.api_field)
        else:
            existing_value = None
        if isinstance(existing_value, list):
            existing_value = existing_value.copy()
        return existing_value

    def _CreateBasicFlag(self, **kwargs):
        return self._CreateFlag(arg_name=self.arg_name, **kwargs)