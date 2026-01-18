from __future__ import annotations
import contextlib
import dataclasses
import typing
import warnings
import weakref
from contextlib import suppress
from urwid.str_util import calc_text_pos, calc_width
from urwid.text_layout import LayoutSegment, trim_line
from urwid.util import (
class CanvasCache:
    """
    Cache for rendered canvases.  Automatically populated and
    accessed by Widget render() MetaClass magic, cleared by
    Widget._invalidate().

    Stores weakrefs to the canvas objects, so an external class
    must maintain a reference for this cache to be effective.
    At present the Screen classes store the last topmost canvas
    after redrawing the screen, keeping the canvases from being
    garbage collected.

    _widgets[widget] = {(wcls, size, focus): weakref.ref(canvas), ...}
    _refs[weakref.ref(canvas)] = (widget, wcls, size, focus)
    _deps[widget} = [dependent_widget, ...]
    """
    _widgets: typing.ClassVar[dict[Widget, dict[tuple[type[Widget], tuple[int, int] | tuple[int] | tuple[()], bool], weakref.ReferenceType]]] = {}
    _refs: typing.ClassVar[dict[weakref.ReferenceType, tuple[Widget, type[Widget], tuple[int, int] | tuple[int] | tuple[()], bool]]] = {}
    _deps: typing.ClassVar[dict[Widget, list[Widget]]] = {}
    hits = 0
    fetches = 0
    cleanups = 0

    @classmethod
    def store(cls, wcls, canvas: Canvas) -> None:
        """
        Store a weakref to canvas in the cache.

        wcls -- widget class that contains render() function
        canvas -- rendered canvas with widget_info (widget, size, focus)
        """
        if not canvas.cacheable:
            return
        if not canvas.widget_info:
            raise TypeError("Can't store canvas without widget_info")
        widget, size, focus = canvas.widget_info

        def walk_depends(canv):
            """
            Collect all child widgets for determining who we
            depend on.
            """
            depends = []
            for _x, _y, c, _pos in canv.children:
                if c.widget_info:
                    depends.append(c.widget_info[0])
                elif hasattr(c, 'children'):
                    depends.extend(walk_depends(c))
            return depends
        depends_on = getattr(canvas, 'depends_on', None)
        if depends_on is None and hasattr(canvas, 'children'):
            depends_on = walk_depends(canvas)
        if depends_on:
            for w in depends_on:
                if w not in cls._widgets:
                    return
            for w in depends_on:
                cls._deps.setdefault(w, []).append(widget)
        ref = weakref.ref(canvas, cls.cleanup)
        cls._refs[ref] = (widget, wcls, size, focus)
        cls._widgets.setdefault(widget, {})[wcls, size, focus] = ref

    @classmethod
    def fetch(cls, widget, wcls, size, focus) -> Canvas | None:
        """
        Return the cached canvas or None.

        widget -- widget object requested
        wcls -- widget class that contains render() function
        size, focus -- render() parameters
        """
        cls.fetches += 1
        sizes = cls._widgets.get(widget, None)
        if not sizes:
            return None
        ref = sizes.get((wcls, size, focus), None)
        if not ref:
            return None
        canv = ref()
        if canv:
            cls.hits += 1
        return canv

    @classmethod
    def invalidate(cls, widget):
        """
        Remove all canvases cached for widget.
        """
        with contextlib.suppress(KeyError):
            for ref in cls._widgets[widget].values():
                with suppress(KeyError):
                    del cls._refs[ref]
            del cls._widgets[widget]
        if widget not in cls._deps:
            return
        dependants = cls._deps.get(widget, [])
        with suppress(KeyError):
            del cls._deps[widget]
        for w in dependants:
            cls.invalidate(w)

    @classmethod
    def cleanup(cls, ref: weakref.ReferenceType) -> None:
        cls.cleanups += 1
        w = cls._refs.get(ref, None)
        del cls._refs[ref]
        if not w:
            return
        widget, wcls, size, focus = w
        sizes = cls._widgets.get(widget, None)
        if not sizes:
            return
        with suppress(KeyError):
            del sizes[wcls, size, focus]
        if not sizes:
            with contextlib.suppress(KeyError):
                del cls._widgets[widget]
                del cls._deps[widget]

    @classmethod
    def clear(cls) -> None:
        """
        Empty the cache.
        """
        cls._widgets = {}
        cls._refs = {}
        cls._deps = {}