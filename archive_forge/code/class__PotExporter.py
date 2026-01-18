import inspect
import os
import sys
import breezy
from . import commands as _mod_commands
from . import errors, help_topics, option
from . import plugin as _mod_plugin
from .i18n import gettext
from .trace import mutter, note
class _PotExporter:
    """Write message details to output stream in .pot file format"""

    def __init__(self, outf, include_duplicates=False):
        self.outf = outf
        if include_duplicates:
            self._msgids = None
        else:
            self._msgids = set()
        self._module_contexts = {}

    def poentry(self, path, lineno, s, comment=None):
        if self._msgids is not None:
            if s in self._msgids:
                return
            self._msgids.add(s)
        if comment is None:
            comment = ''
        else:
            comment = '# %s\n' % comment
        mutter('Exporting msg %r at line %d in %r', s[:20], lineno, path)
        line = '#: {path}:{lineno}\n{comment}msgid {msg}\nmsgstr ""\n\n'.format(path=path, lineno=lineno, comment=comment, msg=_normalize(s))
        self.outf.write(line)

    def poentry_in_context(self, context, string, comment=None):
        context = context.from_string(string)
        self.poentry(context.path, context.lineno, string, comment)

    def poentry_per_paragraph(self, path, lineno, msgid, include=None):
        paragraphs = msgid.split('\n\n')
        if include is not None:
            paragraphs = filter(include, paragraphs)
        for p in paragraphs:
            self.poentry(path, lineno, p)
            lineno += p.count('\n') + 2

    def get_context(self, obj):
        module = inspect.getmodule(obj)
        try:
            context = self._module_contexts[module.__name__]
        except KeyError:
            context = _ModuleContext.from_module(module)
            self._module_contexts[module.__name__] = context
        if inspect.isclass(obj):
            context = context.from_class(obj)
        return context