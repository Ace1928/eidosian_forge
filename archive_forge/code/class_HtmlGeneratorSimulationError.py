from __future__ import annotations
import html
import typing
from urwid import str_util
from urwid.event_loop import ExitMainLoop
from urwid.util import get_encoding
from .common import AttrSpec, BaseScreen
class HtmlGeneratorSimulationError(Exception):
    pass