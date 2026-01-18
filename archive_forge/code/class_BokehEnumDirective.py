from __future__ import annotations
import logging  # isort:skip
import importlib
import textwrap
from docutils.parsers.rst.directives import unchanged
from sphinx.errors import SphinxError
from . import PARALLEL_SAFE
from .bokeh_directive import BokehDirective
from .templates import ENUM_DETAIL
class BokehEnumDirective(BokehDirective):
    has_content = True
    required_arguments = 1
    option_spec = {'module': unchanged, 'noindex': lambda x: True}

    def run(self):
        enum_name = self.arguments[0]
        module_name = self.options['module']
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            raise SphinxError(f'Could not generate reference docs for {enum_name!r}: could not import module {module_name}')
        enum = getattr(module, enum_name, None)
        fullrepr = repr(enum)
        if len(fullrepr) > 180:
            shortrepr = f'{fullrepr[:40]} .... {fullrepr[-40:]}'
            fullrepr = _wrapper.wrap(fullrepr)
        else:
            shortrepr = fullrepr
            fullrepr = None
        rst_text = ENUM_DETAIL.render(name=enum_name, module=self.options['module'], noindex=self.options.get('noindex', False), content=self.content, shortrepr=shortrepr, fullrepr=fullrepr)
        return self.parse(rst_text, '<bokeh-enum>')