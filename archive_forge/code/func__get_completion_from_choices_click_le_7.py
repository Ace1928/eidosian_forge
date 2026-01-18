from __future__ import unicode_literals
import os
from glob import iglob
import click
from prompt_toolkit.completion import Completion, Completer
from .utils import _resolve_context, split_arg_string
def _get_completion_from_choices_click_le_7(self, param, incomplete):
    if not getattr(param.type, 'case_sensitive', True):
        incomplete = incomplete.lower()
        return [Completion(text_type(choice), -len(incomplete), display=text_type(repr(choice) if ' ' in choice else choice)) for choice in param.type.choices if choice.lower().startswith(incomplete)]
    else:
        return [Completion(text_type(choice), -len(incomplete), display=text_type(repr(choice) if ' ' in choice else choice)) for choice in param.type.choices if choice.startswith(incomplete)]