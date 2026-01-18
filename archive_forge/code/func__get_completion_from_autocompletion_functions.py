from __future__ import unicode_literals
import os
from glob import iglob
import click
from prompt_toolkit.completion import Completion, Completer
from .utils import _resolve_context, split_arg_string
def _get_completion_from_autocompletion_functions(self, param, autocomplete_ctx, args, incomplete):
    param_choices = []
    if HAS_CLICK_V8:
        autocompletions = param.shell_complete(autocomplete_ctx, incomplete)
    else:
        autocompletions = param.autocompletion(autocomplete_ctx, args, incomplete)
    for autocomplete in autocompletions:
        if isinstance(autocomplete, tuple):
            param_choices.append(Completion(text_type(autocomplete[0]), -len(incomplete), display_meta=autocomplete[1]))
        elif HAS_CLICK_V8 and isinstance(autocomplete, click.shell_completion.CompletionItem):
            param_choices.append(Completion(text_type(autocomplete.value), -len(incomplete)))
        else:
            param_choices.append(Completion(text_type(autocomplete), -len(incomplete)))
    return param_choices