import argparse
import functools
import os
import pathlib
import re
import sys
import textwrap
from types import ModuleType
from typing import (
from requests.structures import CaseInsensitiveDict
import gitlab.config
from gitlab.base import RESTObject
class VerticalHelpFormatter(argparse.HelpFormatter):

    def format_help(self) -> str:
        result = super().format_help()
        output = ''
        indent = self._indent_increment * ' '
        for line in result.splitlines(keepends=True):
            if line.strip().startswith('{'):
                choice_string, help_string = line.split('}', 1)
                choice_list = choice_string.strip(' {').split(',')
                help_string = help_string.strip()
                if help_string:
                    help_indent = len(max(choice_list, key=len)) * ' '
                    choice_list.append(f'{help_indent} {help_string}')
                choices = '\n'.join(choice_list)
                line = f'{textwrap.indent(choices, indent)}\n'
            output += line
        return output