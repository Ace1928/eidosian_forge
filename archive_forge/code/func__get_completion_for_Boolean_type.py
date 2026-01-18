from __future__ import unicode_literals
import os
from glob import iglob
import click
from prompt_toolkit.completion import Completion, Completer
from .utils import _resolve_context, split_arg_string
def _get_completion_for_Boolean_type(self, param, incomplete):
    return [Completion(text_type(k), -len(incomplete), display_meta=text_type('/'.join(v))) for k, v in {'true': ('1', 'true', 't', 'yes', 'y', 'on'), 'false': ('0', 'false', 'f', 'no', 'n', 'off')}.items() if any((i.startswith(incomplete) for i in v))]