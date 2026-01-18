import asyncio
import sys
from bokeh.document import Document
from bokeh.embed.elements import script_for_render_items
from bokeh.embed.util import standalone_docs_json_and_render_items
from bokeh.embed.wrappers import wrap_in_script_tag
from panel.io.pyodide import _link_docs
from panel.pane import panel as as_panel
from .core.dimension import LabelledData
from .core.options import Store
from .util import extension as _extension
class pyodide_extension(_extension):
    _loaded = False

    def __call__(self, *args, **params):
        super().__call__(*args, **params)
        if not self._loaded:
            Store.output_settings.initialize(list(Store.renderers.keys()))
            Store.set_display_hook('html+js', LabelledData, render_html)
            Store.set_display_hook('png', LabelledData, render_png)
            Store.set_display_hook('svg', LabelledData, render_svg)
            pyodide_extension._loaded = True