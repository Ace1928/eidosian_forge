from __future__ import annotations
import logging  # isort:skip
import importlib
import re
import textwrap
from os.path import basename
from sphinx.errors import SphinxError
from . import PARALLEL_SAFE
from .bokeh_directive import BokehDirective
from .templates import JINJA_DETAIL
class BokehJinjaDirective(BokehDirective):
    has_content = True
    required_arguments = 1
    option_spec = {'noindex': lambda x: True}

    def run(self):
        template_path = self.arguments[0]
        module_path, template_name = template_path.rsplit('.', 1)
        try:
            module = importlib.import_module(module_path)
        except ImportError:
            SphinxError(f'Unable to import Bokeh template module: {module_path}')
        template = getattr(module, template_name, None)
        if template is None:
            SphinxError(f'Unable to find Bokeh template: {template_path}')
        template_text = open(template.filename).read()
        m = _DOCPAT.match(template_text)
        doc = m.group(1) if m else None
        filename = basename(template.filename)
        rst_text = JINJA_DETAIL.render(name=template_name, module=module_path, objrepr=repr(template), noindex=self.options.get('noindex', False), doc='' if doc is None else textwrap.dedent(doc), filename=filename, template_text=_DOCPAT.sub('', template_text))
        return self.parse(rst_text, '<bokeh-jinja>')