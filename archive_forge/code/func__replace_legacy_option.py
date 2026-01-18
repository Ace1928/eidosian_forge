from __future__ import unicode_literals
import optparse
import sys
import six
from pybtex import __version__, errors
from pybtex.plugin import enumerate_plugin_names, find_plugin
from pybtex.textutils import add_period
def _replace_legacy_option(self, arg):
    try:
        unicode_arg = arg if isinstance(arg, six.text_type) else arg.decode('ASCII')
    except UnicodeDecodeError:
        return arg
    if unicode_arg.split('=', 1)[0] in self.legacy_options:
        return type(arg)('-') + arg
    else:
        return arg