from __future__ import annotations
from sphinx.util import logging  # isort:skip
import re
import warnings
from os import getenv
from os.path import basename, dirname, join
from uuid import uuid4
from docutils import nodes
from docutils.parsers.rst.directives import choice, flag
from sphinx.errors import SphinxError
from sphinx.util import copyfile, ensuredir
from sphinx.util.display import status_iterator
from sphinx.util.nodes import set_source_info
from bokeh.document import Document
from bokeh.embed import autoload_static
from bokeh.model import Model
from bokeh.util.warnings import BokehDeprecationWarning
from . import PARALLEL_SAFE
from .bokeh_directive import BokehDirective
from .example_handler import ExampleHandler
from .util import _REPO_TOP, get_sphinx_resources
class BokehPlotDirective(BokehDirective):
    has_content = True
    optional_arguments = 2
    option_spec = {'process-docstring': lambda x: flag(x) is None, 'source-position': lambda x: choice(x, ('below', 'above', 'none')), 'linenos': lambda x: flag(x) is None}

    def run(self):
        if getenv('BOKEH_SPHINX_QUICK') == '1':
            return []
        source, path = self.process_args_or_content()
        dashed_docname = self.env.docname.replace('/', '-')
        js_filename = f'bokeh-content-{uuid4().hex}-{dashed_docname}.js'
        try:
            script_tag, js_path, source, docstring, height_hint = self.process_source(source, path, js_filename)
        except Exception as e:
            raise SphinxError(f'Error generating {js_filename}: \n\n{e}')
        self.env.bokeh_plot_files.add((js_path, dirname(self.env.docname)))
        target_id = f'{dashed_docname}.{basename(js_path)}'
        target = [nodes.target('', '', ids=[target_id])]
        self.process_sampledata(source)
        process_docstring = self.options.get('process-docstring', False)
        intro = self.parse(docstring, '<bokeh-content>') if docstring and process_docstring else []
        above, below = self.process_code_block(source, docstring)
        autoload = [autoload_script(height_hint=height_hint, script_tag=script_tag)]
        return target + intro + above + autoload + below

    def process_code_block(self, source: str, docstring: str | None):
        source_position = self.options.get('source-position', 'below')
        if source_position == 'none':
            return ([], [])
        source = _remove_module_docstring(source, docstring).strip()
        linenos = self.options.get('linenos', False)
        code_block = nodes.literal_block(source, source, language='python', linenos=linenos, classes=[])
        set_source_info(self, code_block)
        if source_position == 'above':
            return ([code_block], [])
        if source_position == 'below':
            return ([], [code_block])

    def process_args_or_content(self):
        if self.arguments and self.content:
            raise SphinxError("bokeh-plot:: directive can't have both args and content")
        if self.content:
            log.debug(f'[bokeh-plot] handling inline content in {self.env.docname!r}')
            path = self.env.bokeh_plot_auxdir
            return ('\n'.join(self.content), path)
        path = self.arguments[0]
        log.debug(f'[bokeh-plot] handling external content in {self.env.docname!r}: {path}')
        if path.startswith('__REPO__/'):
            path = join(_REPO_TOP, path.replace('__REPO__/', ''))
        elif not path.startswith('/'):
            path = join(self.env.app.srcdir, path)
        try:
            with open(path) as f:
                return (f.read(), path)
        except Exception as e:
            raise SphinxError(f'bokeh-plot:: error reading {path!r} for {self.env.docname!r}: {e!r}')

    def process_source(self, source, path, js_filename):
        Model._clear_extensions()
        root, docstring = _evaluate_source(source, path, self.env)
        height_hint = root._sphinx_height_hint()
        js_path = join(self.env.bokeh_plot_auxdir, js_filename)
        js, script_tag = autoload_static(root, RESOURCES, js_filename)
        with open(js_path, 'w') as f:
            f.write(js)
        return (script_tag, js_path, source, docstring, height_hint)

    def process_sampledata(self, source):
        if not hasattr(self.env, 'solved_sampledata'):
            self.env.solved_sampledata = []
        file, lineno = self.get_source_info()
        if '/docs/examples/' in file and file not in self.env.solved_sampledata:
            self.env.solved_sampledata.append(file)
            if not hasattr(self.env, 'all_sampledata_xrefs'):
                self.env.all_sampledata_xrefs = []
            if not hasattr(self.env, 'all_gallery_overview'):
                self.env.all_gallery_overview = []
            self.env.all_gallery_overview.append({'docname': self.env.docname})
            regex = '(:|bokeh\\.)sampledata(:|\\.| import )\\s*(\\w+(\\,\\s*\\w+)*)'
            matches = re.findall(regex, source)
            if matches:
                keywords = set()
                for m in matches:
                    keywords.update(m[2].replace(' ', '').split(','))
                for keyword in keywords:
                    self.env.all_sampledata_xrefs.append({'docname': self.env.docname, 'keyword': keyword})