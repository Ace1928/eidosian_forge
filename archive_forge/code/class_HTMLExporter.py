import base64
import json
import mimetypes
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import jinja2
import markupsafe
from bs4 import BeautifulSoup
from jupyter_core.paths import jupyter_path
from traitlets import Bool, Unicode, default, validate
from traitlets.config import Config
from jinja2.loaders import split_template_path
from nbformat import NotebookNode
from nbconvert.filters.highlight import Highlight2HTML
from nbconvert.filters.markdown_mistune import IPythonRenderer, MarkdownWithMath
from nbconvert.filters.widgetsdatatypefilter import WidgetsDataTypeFilter
from nbconvert.utils.iso639_1 import iso639_1
from .templateexporter import TemplateExporter
class HTMLExporter(TemplateExporter):
    """
    Exports a basic HTML document.  This exporter assists with the export of
    HTML.  Inherit from it if you are writing your own HTML template and need
    custom preprocessors/filters.  If you don't need custom preprocessors/
    filters, just change the 'template_file' config option.
    """
    export_from_notebook = 'HTML'
    anchor_link_text = Unicode('¶', help='The text used as the text for anchor links.').tag(config=True)
    exclude_anchor_links = Bool(False, help='If anchor links should be included or not.').tag(config=True)
    require_js_url = Unicode('https://cdnjs.cloudflare.com/ajax/libs/require.js/2.1.10/require.min.js', help='\n        URL to load require.js from.\n\n        Defaults to loading from cdnjs.\n        ').tag(config=True)
    mathjax_url = Unicode('https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/latest.js?config=TeX-AMS_CHTML-full,Safe', help='\n        URL to load Mathjax from.\n\n        Defaults to loading from cdnjs.\n        ').tag(config=True)
    mermaid_js_url = Unicode('https://cdnjs.cloudflare.com/ajax/libs/mermaid/10.7.0/mermaid.esm.min.mjs', help='\n        URL to load MermaidJS from.\n\n        Defaults to loading from cdnjs.\n        ')
    jquery_url = Unicode('https://cdnjs.cloudflare.com/ajax/libs/jquery/2.0.3/jquery.min.js', help='\n        URL to load jQuery from.\n\n        Defaults to loading from cdnjs.\n        ').tag(config=True)
    jupyter_widgets_base_url = Unicode('https://unpkg.com/', help='URL base for Jupyter widgets').tag(config=True)
    widget_renderer_url = Unicode('', help='Full URL for Jupyter widgets').tag(config=True)
    html_manager_semver_range = Unicode('*', help='Semver range for Jupyter widgets HTML manager').tag(config=True)

    @default('file_extension')
    def _file_extension_default(self):
        return '.html'

    @default('template_name')
    def _template_name_default(self):
        return 'lab'
    theme = Unicode('light', help='Template specific theme(e.g. the name of a JupyterLab CSS theme distributed as prebuilt extension for the lab template)').tag(config=True)
    sanitize_html = Bool(False, help='Whether the HTML in Markdown cells and cell outputs should be sanitized.This should be set to True by nbviewer or similar tools.').tag(config=True)
    skip_svg_encoding = Bool(False, help='Whether the svg to image data attribute encoding should occur').tag(config=True)
    embed_images = Bool(False, help='Whether or not to embed images as base64 in markdown cells.').tag(config=True)
    output_mimetype = 'text/html'

    @property
    def default_config(self):
        c = Config({'NbConvertBase': {'display_data_priority': ['application/vnd.jupyter.widget-view+json', 'application/javascript', 'text/html', 'text/markdown', 'image/svg+xml', 'text/vnd.mermaid', 'text/latex', 'image/png', 'image/jpeg', 'text/plain']}, 'HighlightMagicsPreprocessor': {'enabled': True}})
        if super().default_config:
            c2 = super().default_config.copy()
            c2.merge(c)
            c = c2
        return c
    language_code = Unicode('en', help='Language code of the content, should be one of the ISO639-1').tag(config=True)

    @validate('language_code')
    def _valid_language_code(self, proposal):
        if self.language_code not in iso639_1:
            self.log.warning('"%s" is not an ISO 639-1 language code. It has been replaced by the default value "en".', self.language_code)
            return proposal['trait'].default_value
        return proposal['value']

    @contextfilter
    def markdown2html(self, context, source):
        """Markdown to HTML filter respecting the anchor_link_text setting"""
        cell = context.get('cell', {})
        attachments = cell.get('attachments', {})
        path = context.get('resources', {}).get('metadata', {}).get('path', '')
        renderer = IPythonRenderer(escape=False, attachments=attachments, embed_images=self.embed_images, path=path, anchor_link_text=self.anchor_link_text, exclude_anchor_links=self.exclude_anchor_links)
        return MarkdownWithMath(renderer=renderer).render(source)

    def default_filters(self):
        """Get the default filters."""
        yield from super().default_filters()
        yield ('markdown2html', self.markdown2html)

    def from_notebook_node(self, nb: NotebookNode, resources: Optional[Dict[str, Any]]=None, **kw: Any) -> Tuple[str, Dict[str, Any]]:
        """Convert from notebook node."""
        langinfo = nb.metadata.get('language_info', {})
        lexer = langinfo.get('pygments_lexer', langinfo.get('name', None))
        highlight_code = self.filters.get('highlight_code', Highlight2HTML(pygments_lexer=lexer, parent=self))
        resources = self._init_resources(resources)
        filter_data_type = WidgetsDataTypeFilter(notebook_metadata=self._nb_metadata, parent=self, resources=resources)
        self.register_filter('highlight_code', highlight_code)
        self.register_filter('filter_data_type', filter_data_type)
        html, resources = super().from_notebook_node(nb, resources, **kw)
        soup = BeautifulSoup(html, features='html.parser')
        missing_alt = 0
        for elem in soup.select('img:not([alt])'):
            elem.attrs['alt'] = 'No description has been provided for this image'
            missing_alt += 1
        if missing_alt:
            self.log.warning('Alternative text is missing on %s image(s).', missing_alt)
        for elem in soup.select('.jp-Notebook div.jp-Cell-inputWrapper'):
            elem.attrs['tabindex'] = '0'
        for elem in soup.select('.jp-Notebook div.jp-OutputArea-output'):
            elem.attrs['tabindex'] = '0'
        return (str(soup), resources)

    def _init_resources(self, resources):

        def resources_include_css(name):
            env = self.environment
            code = '<style type="text/css">\n%s</style>' % env.loader.get_source(env, name)[0]
            return markupsafe.Markup(code)

        def resources_include_lab_theme(name):
            _, theme_path = find_lab_theme(name)
            with open(theme_path / 'index.css') as file:
                data = file.read()
            for asset in os.listdir(theme_path):
                local_url = f'url({Path(asset).as_posix()})'
                if local_url in data:
                    mime_type = mimetypes.guess_type(asset)[0]
                    with open(theme_path / asset, 'rb') as assetfile:
                        base64_data = base64.b64encode(assetfile.read())
                        base64_str = base64_data.replace(b'\n', b'').decode('ascii')
                        data = data.replace(local_url, f'url(data:{mime_type};base64,{base64_str})')
            code = '<style type="text/css">\n%s</style>' % data
            return markupsafe.Markup(code)

        def resources_include_js(name, module=False):
            """Get the resources include JS for a name. If module=True, import as ES module"""
            env = self.environment
            code = f'<script {('type="module"' if module else '')}>\n{env.loader.get_source(env, name)[0]}</script>'
            return markupsafe.Markup(code)

        def resources_include_url(name):
            """Get the resources include url for a name."""
            env = self.environment
            mime_type, encoding = mimetypes.guess_type(name)
            try:
                data = env.loader.get_source(env, name)[0].encode('utf8')
            except UnicodeDecodeError:
                pieces = split_template_path(name)
                for searchpath in self.template_paths:
                    filename = os.path.join(searchpath, *pieces)
                    if os.path.exists(filename):
                        with open(filename, 'rb') as f:
                            data = f.read()
                            break
                else:
                    msg = f'No file {name!r} found in {searchpath!r}'
                    raise ValueError(msg)
            data = base64.b64encode(data)
            data = data.replace(b'\n', b'').decode('ascii')
            src = f'data:{mime_type};base64,{data}'
            return markupsafe.Markup(src)
        resources = super()._init_resources(resources)
        resources['theme'] = self.theme
        resources['include_css'] = resources_include_css
        resources['include_lab_theme'] = resources_include_lab_theme
        resources['include_js'] = resources_include_js
        resources['include_url'] = resources_include_url
        resources['require_js_url'] = self.require_js_url
        resources['mathjax_url'] = self.mathjax_url
        resources['mermaid_js_url'] = self.mermaid_js_url
        resources['jquery_url'] = self.jquery_url
        resources['jupyter_widgets_base_url'] = self.jupyter_widgets_base_url
        resources['widget_renderer_url'] = self.widget_renderer_url
        resources['html_manager_semver_range'] = self.html_manager_semver_range
        resources['should_sanitize_html'] = self.sanitize_html
        resources['language_code'] = self.language_code
        resources['should_not_encode_svg'] = self.skip_svg_encoding
        return resources