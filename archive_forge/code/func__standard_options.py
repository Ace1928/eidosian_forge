import inspect
import os
import sys
import breezy
from . import commands as _mod_commands
from . import errors, help_topics, option
from . import plugin as _mod_plugin
from .i18n import gettext
from .trace import mutter, note
def _standard_options(exporter):
    OPTIONS = option.Option.OPTIONS
    context = exporter.get_context(option)
    for name in sorted(OPTIONS):
        opt = OPTIONS[name]
        _write_option(exporter, context.from_string(name), opt, 'option')