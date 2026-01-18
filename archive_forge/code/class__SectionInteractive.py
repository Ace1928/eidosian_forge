from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
import functools
import os
import re
import sys
import textwrap
from googlecloudsdk.core import argv_utils
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.configurations import named_configs
from googlecloudsdk.core.configurations import properties_file as prop_files_lib
from googlecloudsdk.core.docker import constants as const_lib
from googlecloudsdk.core.resource import resource_printer_types as formats
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import http_proxy_types
from googlecloudsdk.core.util import scaled_integer
from googlecloudsdk.generated_clients.apis import apis_map
import six
class _SectionInteractive(_Section):
    """Contains the properties for the 'interactive' section."""

    def __init__(self):
        super(_SectionInteractive, self).__init__('interactive')
        self.bottom_bindings_line = self._AddBool('bottom_bindings_line', default=True, help_text='If True, display the bottom key bindings line.')
        self.bottom_status_line = self._AddBool('bottom_status_line', default=False, help_text='If True, display the bottom status line.')
        self.completion_menu_lines = self._Add('completion_menu_lines', default=4, help_text='Number of lines in the completion menu.')
        self.context = self._Add('context', default='', help_text='Command context string.')
        self.debug = self._AddBool('debug', default=False, hidden=True, help_text='If True, enable the debugging display.')
        self.fixed_prompt_position = self._Add('fixed_prompt_position', default=False, help_text='If True, display the prompt at the same position.')
        self.help_lines = self._Add('help_lines', default=10, help_text='Maximum number of help snippet lines.')
        self.hidden = self._AddBool('hidden', default=False, help_text='If True, expose hidden commands/flags.')
        self.justify_bottom_lines = self._AddBool('justify_bottom_lines', default=False, help_text='If True, left- and right-justify bottom toolbar lines.')
        self.manpage_generator = self._Add('manpage_generator', default=True, help_text='If True, use the manpage CLI tree generator for unsupported commands.')
        self.multi_column_completion_menu = self._AddBool('multi_column_completion_menu', default=False, help_text='If True, display the completions as a multi-column menu.')
        self.obfuscate = self._AddBool('obfuscate', default=False, hidden=True, help_text='If True, obfuscate status PII.')
        self.prompt = self._Add('prompt', default='$ ', help_text='Command prompt string.')
        self.show_help = self._AddBool('show_help', default=True, help_text='If True, show help as command args are being entered.')
        self.suggest = self._AddBool('suggest', default=False, help_text='If True, add command line suggestions based on history.')