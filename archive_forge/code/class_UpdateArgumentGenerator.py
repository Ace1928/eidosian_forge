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
class UpdateArgumentGenerator(six.with_metaclass(abc.ABCMeta, object)):
    """Update flag generator.

  To use this base class, provide required methods for parsing
  (GetArgFromNamespace and GetFieldValueFromNamespace) and override
  the flags that are needed to update the value. For example, if argument
  group requires a set flag, we would override the `set_arg` property and
  ApplySetFlag method.
  """

    def _GetTextFormatOfEmptyValue(self, value):
        if value:
            return value
        if isinstance(value, dict):
            return 'empty map'
        if isinstance(value, list):
            return 'empty list'
        if value is None:
            return 'null'
        return value

    def _CreateFlag(self, arg_name, flag_prefix=None, flag_type=None, action=None, metavar=None, help_text=None):
        """Creates a flag.

    Args:
      arg_name: str, root name of the arg
      flag_prefix: Prefix | None, prefix for the flag name
      flag_type: func, type that flag is used to convert user input
      action: str, flag action
      metavar: str, user specified metavar for flag
      help_text: str, flag help text

    Returns:
      base.Argument with correct params
    """
        flag_name = arg_utils.GetFlagName(arg_name, flag_prefix and flag_prefix.value)
        arg = base.Argument(flag_name, action=action, help=help_text)
        if action == 'store_true':
            return arg
        arg.kwargs['type'] = flag_type
        if (flag_metavar := arg_utils.GetMetavar(metavar, flag_type, flag_name)):
            arg.kwargs['metavar'] = flag_metavar
        return arg

    @property
    def set_arg(self):
        """Flag that sets field to specifed value."""
        return None

    @property
    def clear_arg(self):
        """Flag that clears field."""
        return None

    @property
    def update_arg(self):
        """Flag that updates value if part of existing field."""
        return None

    @property
    def remove_arg(self):
        """Flag that removes value if part of existing field."""
        return None

    def Generate(self, additional_flags=None):
        """Returns ArgumentGroup with all flags specified in generator.

    ArgumentGroup is returned where the set flag is mutually exclusive with
    the rest of the update flags. In addition, remove and clear flags are
    mutually exclusive. The following combinations are allowed

    # sets the foo value to value1,value2
    {command} --foo=value1,value2

    # adds values value3
    {command} --add-foo=value3

    # clears values and sets foo to value4,value5
    {command} --add-foo=value4,value5 --clear

    # removes value4 and adds value6
    {command} --add-foo=value6 --remove-foo=value4

    # removes value6 and then re-adds it
    {command} --add-foo=value6 --remove-foo=value6

    Args:
      additional_flags: [base.Argument], list of additional arguments needed
        to udpate the value

    Returns:
      base.ArgumentGroup, argument group containing flags
    """
        base_group = base.ArgumentGroup(mutex=True, required=False, hidden=self.is_hidden, help='Update {}.'.format(self.arg_name))
        if self.set_arg:
            base_group.AddArgument(self.set_arg)
        update_group = base.ArgumentGroup(required=False)
        if self.update_arg:
            update_group.AddArgument(self.update_arg)
        clear_group = base.ArgumentGroup(mutex=True, required=False)
        if self.clear_arg:
            clear_group.AddArgument(self.clear_arg)
        if self.remove_arg:
            clear_group.AddArgument(self.remove_arg)
        if clear_group.arguments:
            update_group.AddArgument(clear_group)
        if update_group.arguments:
            base_group.AddArgument(update_group)
        if not additional_flags:
            return base_group
        wrapper_group = base.ArgumentGroup(required=False, hidden=self.is_hidden, help='All arguments needed to update {}.'.format(self.arg_name))
        wrapper_group.AddArgument(base_group)
        for arg in additional_flags:
            wrapper_group.AddArgument(arg)
        return wrapper_group

    @abc.abstractmethod
    def GetArgFromNamespace(self, namespace, arg):
        """Retrieves namespace value associated with flag.

    Args:
      namespace: The parsed command line argument namespace.
      arg: base.Argument, used to get namespace value

    Returns:
      value parsed from namespace
    """
        pass

    @abc.abstractmethod
    def GetFieldValueFromMessage(self, existing_message):
        """Retrieves existing field from message.

    Args:
      existing_message: apitools message we need to get field value from

    Returns:
      field value from apitools message
    """
        pass

    def ApplySetFlag(self, existing_val, unused_set_val):
        """Updates result to new value (No-op: implementation in subclass)."""
        return existing_val

    def ApplyClearFlag(self, existing_val, unused_clear_flag):
        """Clears existing value (No-op: implementation in subclass)."""
        return existing_val

    def ApplyRemoveFlag(self, existing_val, unused_remove_val):
        """Removes existing value (No-op: implementation in subclass)."""
        return existing_val

    def ApplyUpdateFlag(self, existing_val, unused_update_val):
        """Updates existing value (No-op: implementation in subclass)."""
        return existing_val

    def Parse(self, namespace, existing_message):
        """Parses update flags from namespace and returns updated message field.

    Args:
      namespace: The parsed command line argument namespace.
      existing_message: Apitools message that exists for given resource.

    Returns:
      Modified existing apitools message field.
    """
        result = self.GetFieldValueFromMessage(existing_message)
        set_value, clear_value, remove_value, update_value = (self.GetArgFromNamespace(namespace, self.set_arg), self.GetArgFromNamespace(namespace, self.clear_arg), self.GetArgFromNamespace(namespace, self.remove_arg), self.GetArgFromNamespace(namespace, self.update_arg))
        result = self.ApplyClearFlag(result, clear_value)
        result = self.ApplyRemoveFlag(result, remove_value)
        result = self.ApplySetFlag(result, set_value)
        result = self.ApplyUpdateFlag(result, update_value)
        return result