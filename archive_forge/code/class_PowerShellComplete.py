import os
import re
import sys
from typing import Any, Dict, List, Tuple
import click
import click.parser
import click.shell_completion
from ._completion_shared import (
class PowerShellComplete(click.shell_completion.ShellComplete):
    name = Shells.powershell.value
    source_template = COMPLETION_SCRIPT_POWER_SHELL

    def source_vars(self) -> Dict[str, Any]:
        return {'complete_func': self.func_name, 'autocomplete_var': self.complete_var, 'prog_name': self.prog_name}

    def get_completion_args(self) -> Tuple[List[str], str]:
        completion_args = os.getenv('_TYPER_COMPLETE_ARGS', '')
        incomplete = os.getenv('_TYPER_COMPLETE_WORD_TO_COMPLETE', '')
        cwords = click.parser.split_arg_string(completion_args)
        args = cwords[1:]
        return (args, incomplete)

    def format_completion(self, item: click.shell_completion.CompletionItem) -> str:
        return f'{item.value}:::{item.help or ' '}'