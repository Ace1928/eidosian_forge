from __future__ import annotations
import logging  # isort:skip
from typing import TYPE_CHECKING
from bokeh.application.handlers.code_runner import CodeRunner
from bokeh.application.handlers.handler import Handler
from bokeh.core.types import PathLike
from bokeh.io.doc import curdoc, set_curdoc
A stripped-down handler similar to CodeHandler but that does
    some appropriate monkeypatching.

    