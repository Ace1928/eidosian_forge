from __future__ import unicode_literals
import os
from glob import iglob
import click
from prompt_toolkit.completion import Completion, Completer
from .utils import _resolve_context, split_arg_string
def _get_completion_for_Path_types(self, param, args, incomplete):
    if '*' in incomplete:
        return []
    choices = []
    _incomplete = os.path.expandvars(incomplete)
    search_pattern = _incomplete.strip('\'"\t\n\r\x0b ').replace('\\\\', '\\') + '*'
    quote = ''
    if ' ' in _incomplete:
        for i in incomplete:
            if i in ("'", '"'):
                quote = i
                break
    for path in iglob(search_pattern):
        if ' ' in path:
            if quote:
                path = quote + path
            elif IS_WINDOWS:
                path = repr(path).replace('\\\\', '\\')
        elif IS_WINDOWS:
            path = path.replace('\\', '\\\\')
        choices.append(Completion(text_type(path), -len(incomplete), display=text_type(os.path.basename(path.strip('\'"')))))
    return choices