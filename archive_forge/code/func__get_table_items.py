from __future__ import annotations
import inspect
from rich.console import Console
from rich.table import Table
import gradio._simple_templates
import gradio.components
import gradio.layouts
from gradio.blocks import BlockContext
from gradio.components import Component, FormComponent
def _get_table_items(module):
    items = []
    for name in module.__all__:
        gr_cls = getattr(module, name)
        if not (inspect.isclass(gr_cls) and issubclass(gr_cls, (Component, BlockContext))) or name in _IGNORE:
            continue
        tags = []
        if 'Simple' in name or name in _BEGINNER_FRIENDLY:
            tags.append(':seedling::handshake:Beginner Friendly:seedling::handshake:')
        if issubclass(gr_cls, FormComponent):
            tags.append(':pencil::jigsaw:Form Component:pencil::jigsaw:')
        if name in gradio.layouts.__all__:
            tags.append(':triangular_ruler:Layout:triangular_ruler:')
        doc = inspect.getdoc(gr_cls) or 'No description available.'
        doc = doc.split('.')[0]
        if tags:
            doc = f'[{', '.join(tags)}]' + ' ' + doc
        items.append((name, doc))
    return items