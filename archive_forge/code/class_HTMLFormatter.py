import argparse
import pkgutil
import warnings
import types as pytypes
from numba.core import errors
from numba._version import get_versions
from numba.core.registry import cpu_target
from numba.tests.support import captured_stdout
class HTMLFormatter(Formatter):
    """Formatter that outputs HTML
    """

    def escape(self, text):
        import html
        return html.escape(text)

    def title(self, text):
        self.print('<h1>', text, '</h2>')

    def begin_module_section(self, modname):
        self.print('<h2>', modname, '</h2>')
        self.print('<ul>')

    def end_module_section(self):
        self.print('</ul>')

    def write_supported_item(self, modname, itemname, typename, explained, sources, alias):
        self.print('<li>')
        self.print('{}.<b>{}</b>'.format(modname, itemname))
        self.print(': <b>{}</b>'.format(typename))
        self.print('<div><pre>', explained, '</pre></div>')
        self.print('<ul>')
        for tcls, source in sources.items():
            if source:
                self.print('<li>')
                impl = source['name']
                sig = source['sig']
                filename = source['filename']
                lines = source['lines']
                self.print('<p>defined by <b>{}</b>{} at {}:{}-{}</p>'.format(self.escape(impl), self.escape(sig), self.escape(filename), lines[0], lines[1]))
                self.print('<p>{}</p>'.format(self.escape(source['docstring'] or '')))
            else:
                self.print('<li>{}'.format(self.escape(str(tcls))))
            self.print('</li>')
        self.print('</ul>')
        self.print('</li>')

    def write_unsupported_item(self, modname, itemname):
        self.print('<li>')
        self.print('{}.<b>{}</b>: UNSUPPORTED'.format(modname, itemname))
        self.print('</li>')

    def write_statistic(self, stats):
        self.print('<p>{}</p>'.format(stats.describe()))