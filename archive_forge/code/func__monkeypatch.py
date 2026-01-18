from __future__ import annotations
import logging  # isort:skip
from typing import TYPE_CHECKING
from bokeh.application.handlers.code_runner import CodeRunner
from bokeh.application.handlers.handler import Handler
from bokeh.core.types import PathLike
from bokeh.io.doc import curdoc, set_curdoc
def _monkeypatch(self):

    def _pass(*args, **kw):
        pass

    def _add_root(obj, *args, **kw):
        curdoc().add_root(obj)

    def _curdoc(*args, **kw):
        return curdoc()
    import bokeh.io as io
    import bokeh.plotting as p
    mods = [io, p]
    old_io = {}
    for f in self._output_funcs + self._io_funcs:
        old_io[f] = getattr(io, f)
    for mod in mods:
        for f in self._output_funcs:
            setattr(mod, f, _pass)
        for f in self._io_funcs:
            setattr(mod, f, _add_root)
    import bokeh.document as d
    old_doc = d.Document
    d.Document = _curdoc
    return (old_io, old_doc)