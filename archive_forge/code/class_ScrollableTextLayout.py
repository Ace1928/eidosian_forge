from __future__ import annotations
from typing import Type, Optional, TYPE_CHECKING, Tuple
from pyglet import graphics
from pyglet.customtypes import AnchorY, AnchorX
from pyglet.gl import glActiveTexture, GL_TEXTURE0, glBindTexture, glEnable, GL_BLEND, glBlendFunc, GL_SRC_ALPHA, \
from pyglet.text.layout.base import TextLayout
class ScrollableTextLayout(TextLayout):
    """Display text in a scrollable viewport.

    This class does not display a scrollbar or handle scroll events; it merely
    clips the text that would be drawn in :py:func:`~pyglet.text.layout.TextLayout`
    to the bounds of the layout given by `x`, `y`, `width` and `height`;
    and offsets the text by a scroll offset.

    Use `view_x` and `view_y` to scroll the text within the viewport.
    """
    group_class: Type[ScrollableTextLayoutGroup] = ScrollableTextLayoutGroup
    decoration_class: Type[ScrollableTextDecorationGroup] = ScrollableTextDecorationGroup
    _translate_x: int = 0
    _translate_y: int = 0

    def __init__(self, document: AbstractDocument, width: int, height: int, x: float=0, y: float=0, z: float=0, anchor_x: AnchorX='left', anchor_y: AnchorY='bottom', rotation: float=0, multiline: bool=False, dpi: Optional[float]=None, batch: Optional[Batch]=None, group: Optional[graphics.Group]=None, program: Optional[ShaderProgram]=None, wrap_lines: bool=True) -> None:
        if width is None or height is None:
            raise Exception('Invalid size. ScrollableTextLayout width or height cannot be None.')
        super().__init__(document, width, height, x, y, z, anchor_x, anchor_y, rotation, multiline, dpi, batch, group, program, wrap_lines)

    def _update_scissor_area(self) -> None:
        if not self.document.text:
            return
        area = (self.left, self.bottom, self._width, self._height)
        for group in self.group_cache.values():
            group.scissor_area = area
        self.background_decoration_group.scissor_area = area
        self.foreground_decoration_group.scissor_area = area

    def _update(self) -> None:
        super()._update()
        self._update_scissor_area()

    @property
    def x(self) -> float:
        return self._x

    @x.setter
    def x(self, x: float) -> None:
        super()._set_x(x)
        self._update_scissor_area()

    @property
    def y(self) -> float:
        return self._y

    @y.setter
    def y(self, y: float) -> None:
        super()._set_y(y)
        self._update_scissor_area()

    @property
    def z(self) -> float:
        return self._z

    @z.setter
    def z(self, z: float) -> None:
        super()._set_z(z)
        self._update_scissor_area()

    @property
    def position(self) -> Tuple[float, float, float]:
        return (self._x, self._y, self._z)

    @position.setter
    def position(self, position: Tuple[float, float, float]) -> None:
        super()._set_position(position)
        self._update_scissor_area()

    @property
    def anchor_x(self) -> AnchorX:
        return self._anchor_x

    @anchor_x.setter
    def anchor_x(self, anchor_x: AnchorX) -> None:
        self._anchor_x = anchor_x
        self._update_anchor()
        self._update_scissor_area()
        self._update_view_translation()

    @property
    def anchor_y(self) -> AnchorY:
        return self._anchor_y

    @anchor_y.setter
    def anchor_y(self, anchor_y: AnchorY) -> None:
        self._anchor_y = anchor_y
        self._update_anchor()
        self._update_scissor_area()
        self._update_view_translation()

    def _get_bottom_anchor(self) -> float:
        """Returns the anchor for the Y axis from the bottom."""
        height = self._height
        if self._content_valign == 'top':
            offset = min(0, self._height)
        elif self._content_valign == 'bottom':
            offset = 0
        elif self._content_valign == 'center':
            offset = min(0, self._height) // 2
        else:
            assert False, '`content_valign` must be either "top", "bottom", or "center".'
        if self._anchor_y == 'top':
            return -height + offset
        elif self._anchor_y == 'baseline':
            return -height + self._ascent
        elif self._anchor_y == 'bottom':
            return 0
        elif self._anchor_y == 'center':
            if self._line_count == 1 and self._height is None:
                return self._ascent // 2 - self._descent // 4 - height
            else:
                return offset - height // 2
        else:
            assert False, '`anchor_y` must be either "top", "bottom", "center", or "baseline".'

    def _update_view_translation(self) -> None:
        for _vertex_list in self._vertex_lists:
            _vertex_list.view_translation[:] = (-self._translate_x, -self._translate_y, 0) * _vertex_list.count

    @property
    def view_x(self) -> int:
        """Horizontal scroll offset.

            The initial value is 0, and the left edge of the text will touch the left
            side of the layout bounds.  A positive value causes the text to "scroll"
            to the right.  Values are automatically clipped into the range
            ``[0, content_width - width]``

            :type: int
        """
        return self._translate_x

    @view_x.setter
    def view_x(self, view_x: int) -> None:
        translation = max(0, min(self._content_width - self._width, view_x))
        if translation != self._translate_x:
            self._translate_x = translation
            self._update_view_translation()

    @property
    def view_y(self) -> int:
        """Vertical scroll offset.

            The initial value is 0, and the top of the text will touch the top of the
            layout bounds (unless the content height is less than the layout height,
            in which case `content_valign` is used).

            A negative value causes the text to "scroll" upwards.  Values outside of
            the range ``[height - content_height, 0]`` are automatically clipped in
            range.

            :type: int
        """
        return self._translate_y

    @view_y.setter
    def view_y(self, view_y: int) -> None:
        translation = min(0, max(self.height - self._content_height, view_y))
        if translation != self._translate_y:
            self._translate_y = translation
            self._update_view_translation()