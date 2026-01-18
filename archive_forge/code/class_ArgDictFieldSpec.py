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
@dataclasses.dataclass(frozen=True)
class ArgDictFieldSpec:
    """Attributes about the fields that make up an ArgDict spec.

  Attributes:
    api_field: The name of the field under the repeated message that the value
      should be put.
    arg_name: The name of the key in the dict.
    field_type: The argparse type of the value of this field.
    required: True if the key is required.
    choices: A static map of choice to value the user types.
  """

    @classmethod
    def FromData(cls, data):
        data_choices = data.get('choices')
        choices = [Choice(d) for d in data_choices] if data_choices else None
        return cls(api_field=data['api_field'], arg_name=data.get('arg_name'), field_type=ParseType(data), required=data.get('required', True), choices=choices)
    api_field: str | None
    arg_name: str | None
    field_type: Callable[[str], Any] | None
    required: bool
    choices: list[Choice] | None

    def ChoiceMap(self):
        return Choice.ToChoiceMap(self.choices)