from __future__ import annotations
import logging  # isort:skip
from typing import TYPE_CHECKING
from bokeh.application.handlers.code_runner import CodeRunner
from bokeh.application.handlers.handler import Handler
from bokeh.core.types import PathLike
from bokeh.io.doc import curdoc, set_curdoc
def _unmonkeypatch(self, old_io, old_doc):
    import bokeh.io as io
    import bokeh.plotting as p
    mods = [io, p]
    for mod in mods:
        for f in old_io:
            setattr(mod, f, old_io[f])
    import bokeh.document as d
    d.Document = old_doc