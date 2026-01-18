from __future__ import annotations
import contextlib
import enum
import typing
from typing_extensions import Protocol, runtime_checkable
from .constants import BOX_SYMBOLS, SHADE_SYMBOLS, Sizing
from .widget_decoration import WidgetDecoration, WidgetError
class Scrollable(WidgetDecoration[WrappedWidget]):

    def sizing(self) -> frozenset[Sizing]:
        return frozenset((Sizing.BOX,))

    def selectable(self) -> bool:
        return True

    def __init__(self, widget: WrappedWidget, force_forward_keypress: bool=False) -> None:
        """Box widget that makes a fixed or flow widget vertically scrollable

        .. note::
            Focusable widgets are handled, including switching focus, but possibly not intuitively,
            depending on the arrangement of widgets.

            When switching focus to a widget that is ouside of the visible part of the original widget,
            the canvas scrolls up/down to the focused widget.

            It would be better to scroll until the next focusable widget is in sight first.
            But for that to work we must somehow obtain a list of focusable rows in the original canvas.
        """
        if not widget.sizing() & frozenset((Sizing.FIXED, Sizing.FLOW)):
            raise ValueError(f'Not a fixed or flow widget: {widget!r}')
        self._trim_top = 0
        self._scroll_action = None
        self._forward_keypress = None
        self._old_cursor_coords = None
        self._rows_max_cached = 0
        self.force_forward_keypress = force_forward_keypress
        super().__init__(widget)

    def render(self, size: tuple[int, int], focus: bool=False) -> CompositeCanvas:
        from urwid import canvas
        maxcol, maxrow = size

        def automove_cursor() -> None:
            ch = 0
            last_hidden = False
            first_visible = False
            for pwi, (w, _o) in enumerate(ow.contents):
                wcanv = w.render((maxcol,))
                wh = wcanv.rows()
                if wh:
                    ch += wh
                if not last_hidden and ch >= self._trim_top:
                    last_hidden = True
                elif last_hidden:
                    if not first_visible:
                        first_visible = True
                    if not w.selectable():
                        continue
                    ow.focus_item = pwi
                    st = None
                    nf = ow.get_focus()
                    if hasattr(nf, 'key_timeout'):
                        st = nf
                    elif hasattr(nf, 'original_widget'):
                        no = nf.original_widget
                        if hasattr(no, 'original_widget'):
                            st = no.original_widget
                        elif hasattr(no, 'key_timeout'):
                            st = no
                    if st and hasattr(st, 'key_timeout') and callable(getattr(st, 'keypress', None)):
                        st.keypress(None, None)
                    break
        ow = self._original_widget
        ow_size = self._get_original_widget_size(size)
        canv_full = ow.render(ow_size, focus)
        canv = canvas.CompositeCanvas(canv_full)
        canv_cols, canv_rows = (canv.cols(), canv.rows())
        if canv_cols <= maxcol:
            pad_width = maxcol - canv_cols
            if pad_width > 0:
                canv.pad_trim_left_right(0, pad_width)
        if canv_rows <= maxrow:
            fill_height = maxrow - canv_rows
            if fill_height > 0:
                canv.pad_trim_top_bottom(0, fill_height)
        if canv_cols <= maxcol and canv_rows <= maxrow:
            return canv
        self._adjust_trim_top(canv, size)
        trim_top = self._trim_top
        trim_end = canv_rows - maxrow - trim_top
        trim_right = canv_cols - maxcol
        if trim_top > 0:
            canv.trim(trim_top)
        if trim_end > 0:
            canv.trim_end(trim_end)
        if trim_right > 0:
            canv.pad_trim_left_right(0, -trim_right)
        if canv.cursor is not None:
            _curscol, cursrow = canv.cursor
            if cursrow >= maxrow or cursrow < 0:
                canv.cursor = None
        if canv.cursor is not None:
            self._forward_keypress = True
        elif canv_full.cursor is not None:
            self._forward_keypress = False
            if getattr(ow, 'automove_cursor_on_scroll', False):
                with contextlib.suppress(Exception):
                    automove_cursor()
        else:
            self._forward_keypress = ow.selectable()
        return canv

    def keypress(self, size: tuple[int, int], key: str) -> str | None:
        from urwid.command_map import Command
        if self._forward_keypress or self.force_forward_keypress:
            ow = self._original_widget
            ow_size = self._get_original_widget_size(size)
            if hasattr(ow, 'get_cursor_coords'):
                self._old_cursor_coords = ow.get_cursor_coords(ow_size)
            key = ow.keypress(ow_size, key)
            if key is None:
                return None
        command_map = self._command_map
        if command_map[key] == Command.UP:
            self._scroll_action = SCROLL_LINE_UP
        elif command_map[key] == Command.DOWN:
            self._scroll_action = SCROLL_LINE_DOWN
        elif command_map[key] == Command.PAGE_UP:
            self._scroll_action = SCROLL_PAGE_UP
        elif command_map[key] == Command.PAGE_DOWN:
            self._scroll_action = SCROLL_PAGE_DOWN
        elif command_map[key] == Command.MAX_LEFT:
            self._scroll_action = SCROLL_TO_TOP
        elif command_map[key] == Command.MAX_RIGHT:
            self._scroll_action = SCROLL_TO_END
        else:
            return key
        self._invalidate()
        return None

    def mouse_event(self, size: tuple[int, int], event: str, button: int, col: int, row: int, focus: bool) -> bool | None:
        ow = self._original_widget
        if hasattr(ow, 'mouse_event'):
            ow_size = self._get_original_widget_size(size)
            row += self._trim_top
            return ow.mouse_event(ow_size, event, button, col, row, focus)
        return False

    def _adjust_trim_top(self, canv: Canvas, size: tuple[int, int]) -> None:
        """Adjust self._trim_top according to self._scroll_action"""
        action = self._scroll_action
        self._scroll_action = None
        _maxcol, maxrow = size
        trim_top = self._trim_top
        canv_rows = canv.rows()
        if trim_top < 0:
            trim_top = canv_rows - maxrow + trim_top + 1
        if canv_rows <= maxrow:
            self._trim_top = 0
            return

        def ensure_bounds(new_trim_top: int) -> int:
            return max(0, min(canv_rows - maxrow, new_trim_top))
        if action == SCROLL_LINE_UP:
            self._trim_top = ensure_bounds(trim_top - 1)
        elif action == SCROLL_LINE_DOWN:
            self._trim_top = ensure_bounds(trim_top + 1)
        elif action == SCROLL_PAGE_UP:
            self._trim_top = ensure_bounds(trim_top - maxrow + 1)
        elif action == SCROLL_PAGE_DOWN:
            self._trim_top = ensure_bounds(trim_top + maxrow - 1)
        elif action == SCROLL_TO_TOP:
            self._trim_top = 0
        elif action == SCROLL_TO_END:
            self._trim_top = canv_rows - maxrow
        else:
            self._trim_top = ensure_bounds(trim_top)
        if self._old_cursor_coords is not None and self._old_cursor_coords != canv.cursor and (canv.cursor is not None):
            self._old_cursor_coords = None
            _curscol, cursrow = canv.cursor
            if cursrow < self._trim_top:
                self._trim_top = cursrow
            elif cursrow >= self._trim_top + maxrow:
                self._trim_top = max(0, cursrow - maxrow + 1)

    def _get_original_widget_size(self, size: tuple[int, int]) -> tuple[int] | tuple[()]:
        ow = self._original_widget
        sizing = ow.sizing()
        if Sizing.FLOW in sizing:
            return (size[0],)
        if Sizing.FIXED in sizing:
            return ()
        raise ScrollableError(f'{ow!r} sizing is not supported')

    def get_scrollpos(self, size: tuple[int, int] | None=None, focus: bool=False) -> int:
        """Current scrolling position.

        Lower limit is 0, upper limit is the maximum number of rows with the given maxcol minus maxrow.

        ..note::
            The returned value may be too low or too high if the position has
            changed but the widget wasn't rendered yet.
        """
        return self._trim_top

    def set_scrollpos(self, position: typing.SupportsInt) -> None:
        """Set scrolling position

        If `position` is positive it is interpreted as lines from the top.
        If `position` is negative it is interpreted as lines from the bottom.

        Values that are too high or too low values are automatically adjusted during rendering.
        """
        self._trim_top = int(position)
        self._invalidate()

    def rows_max(self, size: tuple[int, int] | None=None, focus: bool=False) -> int:
        """Return the number of rows for `size`

        If `size` is not given, the currently rendered number of rows is returned.
        """
        if size is not None:
            ow = self._original_widget
            ow_size = self._get_original_widget_size(size)
            sizing = ow.sizing()
            if Sizing.FIXED in sizing:
                self._rows_max_cached = ow.pack(ow_size, focus)[1]
            elif Sizing.FLOW in sizing:
                self._rows_max_cached = ow.rows(ow_size, focus)
            else:
                raise ScrollableError(f'Not a flow/box widget: {self._original_widget!r}')
        return self._rows_max_cached