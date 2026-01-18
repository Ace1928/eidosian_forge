import argparse
import pkgutil
import warnings
import types as pytypes
from numba.core import errors
from numba._version import get_versions
from numba.core.registry import cpu_target
from numba.tests.support import captured_stdout
class ReSTFormatter(Formatter):
    """Formatter that output ReSTructured text format for Sphinx docs.
    """

    def escape(self, text):
        return text

    def title(self, text):
        self.print(text)
        self.print('=' * len(text))
        self.print()

    def begin_module_section(self, modname):
        self.print(modname)
        self.print('-' * len(modname))
        self.print()

    def end_module_section(self):
        self.print()

    def write_supported_item(self, modname, itemname, typename, explained, sources, alias):
        self.print('.. function:: {}.{}'.format(modname, itemname))
        self.print('   :noindex:')
        self.print()
        if alias:
            self.print('   Alias to: ``{}``'.format(alias))
        self.print()
        for tcls, source in sources.items():
            if source:
                impl = source['name']
                sig = source['sig']
                filename = source['filename']
                lines = source['lines']
                source_link = github_url.format(commit=commit, path=filename, firstline=lines[0], lastline=lines[1])
                self.print('   - defined by ``{}{}`` at `{}:{}-{} <{}>`_'.format(impl, sig, filename, lines[0], lines[1], source_link))
            else:
                self.print('   - defined by ``{}``'.format(str(tcls)))
        self.print()

    def write_unsupported_item(self, modname, itemname):
        pass

    def write_statistic(self, stat):
        if stat.supported == 0:
            self.print('This module is not supported.')
        else:
            msg = 'Not showing {} unsupported functions.'
            self.print(msg.format(stat.unsupported))
            self.print()
            self.print(stat.describe())
        self.print()