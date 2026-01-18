from __future__ import annotations
import re
import sys
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Pattern, Union, Optional, List, Any, Tuple, Callable, Iterator, Type, Dict, \
import pyglet
from pyglet import graphics
from pyglet.customtypes import AnchorX, AnchorY, ContentVAlign, HorizontalAlign
from pyglet.font.base import Font, Glyph
from pyglet.gl import GL_TRIANGLES, GL_LINES, glActiveTexture, GL_TEXTURE0, glBindTexture, glEnable, GL_BLEND, \
from pyglet.image import Texture
from pyglet.text import runlist
from pyglet.text.runlist import RunIterator, AbstractRunIterator
class TextLayout:
    """Lay out and display documents.

    This class is intended for displaying documents that do not change
    regularly -- any change will cost some time to lay out the complete
    document again and regenerate all vertex lists.

    The benefit of this class is that texture state is shared between
    all layouts of this class.  The time to draw one :py:func:`~pyglet.text.layout.TextLayout` may be
    roughly the same as the time to draw one :py:class:`~pyglet.text.layout.IncrementalTextLayout`; but
    drawing ten :py:func:`~pyglet.text.layout.TextLayout` objects in one batch is much faster than drawing
    ten incremental or scrollable text layouts.

    :py:func:`~pyglet.text.Label` and :py:func:`~pyglet.text.HTMLLabel` provide a convenient interface to this class.

    :Ivariables:
        `content_width` : int
            Calculated width of the text in the layout.  This may overflow
            the desired width if word-wrapping failed.
        `content_height` : int
            Calculated height of the text in the layout.
        `group_class` : `~pyglet.graphics.Group`
            Top-level rendering group.
        `background_decoration_group` : `~pyglet.graphics.Group`
            Rendering group for background color.
        `foreground_decoration_group` : `~pyglet.graphics.Group`
            Rendering group for glyph underlines.

    """
    _vertex_lists: List[_LayoutVertexList]
    _boxes: List[_AbstractBox]
    group_cache: Dict[Texture, graphics.Group]
    _document: Optional[AbstractDocument] = None
    _update_enabled: bool = True
    _own_batch: bool = False
    group_class: Type[TextLayoutGroup] = TextLayoutGroup
    decoration_class: Type[TextDecorationGroup] = TextDecorationGroup
    _ascent: float = 0
    _descent: float = 0
    _line_count: int = 0
    _anchor_left: float = 0
    _anchor_bottom: float = 0
    _x: float
    _y: float
    _z: float
    _rotation: float = 0
    _width: Optional[int] = None
    _height: Optional[int] = None
    _anchor_x: AnchorX = 'left'
    _anchor_y: AnchorY = 'bottom'
    _content_valign: ContentVAlign = 'top'
    _multiline: bool = False
    _visible: bool = True

    def __init__(self, document: AbstractDocument, width: Optional[int]=None, height: Optional[int]=None, x: float=0, y: float=0, z: float=0, anchor_x: AnchorX='left', anchor_y: AnchorY='bottom', rotation: float=0, multiline: bool=False, dpi: Optional[float]=None, batch: Batch=None, group: Optional[graphics.Group]=None, program: Optional[ShaderProgram]=None, wrap_lines: bool=True, init_document: bool=True) -> None:
        """Create a text layout.

        :Parameters:
            `document` : `AbstractDocument`
                Document to display.
            `x` : int
                X coordinate of the label.
            `y` : int
                Y coordinate of the label.
            `z` : int
                Z coordinate of the label.
            `width` : int
                Width of the layout in pixels, or None
            `height` : int
                Height of the layout in pixels, or None
            `anchor_x` : str
                Anchor point of the X coordinate: one of ``"left"``,
                ``"center"`` or ``"right"``.
            `anchor_y` : str
                Anchor point of the Y coordinate: one of ``"bottom"``,
                ``"baseline"``, ``"center"`` or ``"top"``.
            `rotation`: float
                The amount to rotate the label in degrees. A positive amount
                will be a clockwise rotation, negative values will result in
                counter-clockwise rotation.
            `multiline` : bool
                If False, newline and paragraph characters are ignored, and
                text is not word-wrapped.
                If True, text is wrapped only if the `wrap_lines` is True.
            `dpi` : float
                Font resolution; defaults to 96.
            `batch` : `~pyglet.graphics.Batch`
                Optional graphics batch to add this layout to.
            `group` : `~pyglet.graphics.Group`
                Optional Group to parent all internal Groups that this text
                layout uses.  Note that layouts with the same Groups will
                be rendered simultaneously in a Batch.
            `program` : `~pyglet.graphics.shader.ShaderProgram`
                Optional graphics shader to use. Will affect all glyphs.
            `wrap_lines` : bool
                If True and `multiline` is True, the text is word-wrapped using
                the specified width.
            `init_document` : bool
                If True the document will be initialized. If subclassing then
                you may want to avoid duplicate initializations by changing
                to False.
        """
        self._x = x
        self._y = y
        self._z = z
        self._rotation = rotation
        self._anchor_x = anchor_x
        self._anchor_y = anchor_y
        self._content_width = 0
        self._content_height = 0
        self._user_group = group
        self._vertex_lists = []
        self._boxes = []
        self.group_cache = {}
        self._initialize_groups()
        if batch is None:
            batch = graphics.Batch()
            self._own_batch = True
        self._batch = batch
        self._width = width
        self._height = height
        self._multiline = multiline
        self._program = program or get_default_layout_shader()
        self._wrap_lines_flag = wrap_lines
        self._wrap_lines_invariant()
        self._dpi = dpi or 96
        self._set_document(document)
        if init_document:
            self._init_document()

    @property
    def _flow_glyphs(self) -> Callable:
        if self._multiline:
            return self._flow_glyphs_wrap
        else:
            return self._flow_glyphs_single_line

    def _initialize_groups(self) -> None:
        decoration_shader = get_default_decoration_shader()
        self.background_decoration_group = self.decoration_class(decoration_shader, order=0, parent=self._user_group)
        self.foreground_decoration_group = self.decoration_class(decoration_shader, order=2, parent=self._user_group)

    @property
    def group(self) -> Optional[graphics.Group]:
        return self._user_group

    @group.setter
    def group(self, group: graphics.Group) -> None:
        self._user_group = group
        self._initialize_groups()
        self.group_cache.clear()
        self._update()

    @property
    def dpi(self) -> float:
        """Get DPI used by this layout.

        :type: float
        """
        return self._dpi

    @property
    def document(self) -> AbstractDocument:
        """Document to display.

         For :py:class:`~pyglet.text.layout.IncrementalTextLayout` it is
         far more efficient to modify a document in-place than to replace
         the document instance on the layout.

         :type: `AbstractDocument`
         """
        return self._document

    @document.setter
    def document(self, document: AbstractDocument) -> None:
        self._set_document(document)
        self._init_document()

    def _set_document(self, document: AbstractDocument) -> None:
        if self._document:
            self._document.remove_handlers(self)
            self._uninit_document()
        document.push_handlers(self)
        self._document = document

    @property
    def batch(self) -> Batch:
        """The Batch that this Layout is assigned to.

        If no Batch is assigned, an internal Batch is
        created and used.

        :type: :py:class:`~pyglet.graphics.Batch`

        """
        return self._batch

    @batch.setter
    def batch(self, batch: Optional[Batch]) -> None:
        if self._batch == batch:
            return
        if batch is None:
            self._batch = graphics.Batch()
            self._own_batch = True
            self._update()
        elif batch is not None:
            self._batch = batch
            self._own_batch = False
            self._update()

    @property
    def program(self) -> ShaderProgram:
        """The ShaderProgram that is assigned to this Layout.

        If set, this shader will impact all text layouts except InlineElements.

        :type: :py:class:`~pyglet.graphics.shader.ShaderProgram`

        """
        return self._program

    @program.setter
    def program(self, shader_program: ShaderProgram) -> None:
        if self._program == shader_program:
            return
        self._program = shader_program
        self._update()

    @property
    def x(self) -> float:
        """X coordinate of the layout.

        See also :py:attr:`~pyglet.text.layout.TextLayout.anchor_x`.

        :type: int
        """
        return self._x

    @x.setter
    def x(self, x: float) -> None:
        self._set_x(x)

    def _set_x(self, x: float) -> None:
        self._x = x
        self._update_translation()

    @property
    def y(self) -> float:
        """Y coordinate of the layout.

        See also `anchor_y`.

        :type: int
        """
        return self._y

    @y.setter
    def y(self, y: float) -> None:
        self._set_y(y)

    def _set_y(self, y: float) -> None:
        self._y = y
        self._update_translation()

    @property
    def z(self) -> float:
        """Z coordinate of the layout.

        :type: int
        """
        return self._z

    @z.setter
    def z(self, z: float) -> None:
        self._set_z(z)

    def _set_z(self, z: float) -> None:
        self._z = z
        self._update_translation()

    @property
    def rotation(self) -> float:
        """Rotation of the layout.

        :type: float
        """
        return self._rotation

    @rotation.setter
    def rotation(self, rotation: float) -> None:
        self._set_rotation(rotation)

    def _set_rotation(self, rotation: float) -> None:
        self._rotation = rotation
        self._update_rotation()

    def _update_rotation(self) -> None:
        for box in self._boxes:
            box.update_rotation(self._rotation)

    @property
    def position(self) -> Tuple[float, float, float]:
        """The (X, Y, Z) coordinates of the layout, as a tuple.

        See also :py:attr:`~pyglet.text.layout.TextLayout.anchor_x`,
        and :py:attr:`~pyglet.text.layout.TextLayout.anchor_y`.

        :type: (int, int, int)
        """
        return (self._x, self._y, self._z)

    @position.setter
    def position(self, position: Tuple[float, float, float]) -> None:
        self._set_position(position)

    def _set_position(self, position: Tuple[float, float, float]) -> None:
        self._x, self._y, self._z = position
        self._update_translation()

    def _update_translation(self):
        for box in self._boxes:
            box.update_translation(self._x, self._y, self._z)

    def _update_anchor(self) -> None:
        self._anchor_left = self._get_left_anchor()
        self._anchor_bottom = self._get_bottom_anchor()
        anchor = (self._anchor_left, self._get_top_anchor())
        for box in self._boxes:
            box.update_anchor(*anchor)

    @property
    def visible(self) -> bool:
        """True if the layout will be visible when drawn.

        :type: bool
        """
        return self._visible

    @visible.setter
    def visible(self, value: bool) -> None:
        if value != self._visible:
            self._visible = value
            for box in self._boxes:
                box.update_visibility(value)

    @property
    def content_width(self) -> int:
        """Calculated width of the text in the layout.

        This is the actual width of the text in pixels, not the
        user defined :py:attr:`~ppyglet.text.layout.TextLayout.width`.
        The content width may overflow the layout width if word-wrapping
        is not possible.
        """
        return self._content_width

    @property
    def content_height(self) -> int:
        """The calculated height of the text in the layout.

        This is the actual height of the text in pixels, not the
        user defined :py:attr:`~ppyglet.text.layout.TextLayout.height`.
        """
        return self._content_height

    @property
    def width(self) -> Optional[int]:
        """The defined maximum width of the layout in pixels, or None

        If `multiline` and `wrap_lines` is True, the `width` defines where the
        text will be wrapped. If `multiline` is False or `wrap_lines` is False,
        this property has no effect.

        :type: int
        """
        return self._width

    @width.setter
    def width(self, width: Optional[int]) -> None:
        self._width = width
        self._wrap_lines_invariant()
        self._update()

    @property
    def height(self) -> Optional[int]:
        """The defined maximum height of the layout in pixels, or None

        When `height` is not None, it affects the positioning of the
        text when :py:attr:`~pyglet.text.layout.TextLayout.anchor_y` and
        :py:attr:`~pyglet.text.layout.TextLayout.content_valign` are
        used.

        :type: int
        """
        return self._height

    @height.setter
    def height(self, height: Optional[int]) -> None:
        self._height = height
        self._update()

    @property
    def multiline(self) -> bool:
        """Set if multiline layout is enabled.

        If multiline is False, newline and paragraph characters are ignored and
        text is not word-wrapped.
        If True, the text is word-wrapped only if the `wrap_lines` is True.

        :type: bool
        """
        return self._multiline

    @multiline.setter
    def multiline(self, multiline: bool) -> None:
        self._multiline = multiline
        self._wrap_lines_invariant()
        self._update()

    @property
    def anchor_x(self) -> AnchorX:
        """Horizontal anchor alignment.

        This property determines the meaning of the `x` coordinate.
        It is one of the enumerants:

        ``"left"`` (default)
            The X coordinate gives the position of the left edge of the layout.
        ``"center"``
            The X coordinate gives the position of the center of the layout.
        ``"right"``
            The X coordinate gives the position of the right edge of the layout.

        For the purposes of calculating the position resulting from this
        alignment, the width of the layout is taken to be `width` if `multiline`
        is True and `wrap_lines` is True, otherwise `content_width`.

        :type: str
        """
        return self._anchor_x

    @anchor_x.setter
    def anchor_x(self, anchor_x: AnchorX) -> None:
        self._anchor_x = anchor_x
        self._update_anchor()

    @property
    def anchor_y(self) -> AnchorY:
        """Vertical anchor alignment.

        This property determines the meaning of the `y` coordinate.
        It is one of the enumerants:

        ``"top"``
            The Y coordinate gives the position of the top edge of the layout.
        ``"center"``
            The Y coordinate gives the position of the center of the layout.
        ``"baseline"``
            The Y coordinate gives the position of the baseline of the first
            line of text in the layout.
        ``"bottom"`` (default)
            The Y coordinate gives the position of the bottom edge of the layout.

        For the purposes of calculating the position resulting from this
        alignment, the height of the layout is taken to be the smallest of
        `height` and `content_height`.

        See also `content_valign`.

        :type: str
        """
        return self._anchor_y

    @anchor_y.setter
    def anchor_y(self, anchor_y: AnchorY) -> None:
        self._anchor_y = anchor_y
        self._update_anchor()

    @property
    def content_valign(self) -> ContentVAlign:
        """Vertical alignment of content within larger layout box.

        This property determines how content is positioned within the layout
        box when ``content_height`` is less than ``height``.  It is one
        of the enumerants:

        ``top`` (default)
            Content is aligned to the top of the layout box.
        ``center``
            Content is centered vertically within the layout box.
        ``bottom``
            Content is aligned to the bottom of the layout box.

        This property has no effect when ``content_height`` is greater
        than ``height`` (in which case the content is aligned to the top) or when
        ``height`` is ``None`` (in which case there is no vertical layout box
        dimension).

        :type: str
        """
        return self._content_valign

    @content_valign.setter
    def content_valign(self, content_valign: ContentVAlign) -> None:
        self._content_valign = content_valign
        self._update()

    @property
    def left(self) -> float:
        """
        The x-coordinate of the left side of the layout.

        :type: int
        """
        return self._x + self._anchor_left

    @property
    def right(self) -> float:
        """
        The x-coordinate of the right side of the layout.

        :type: int
        """
        if self._width is None:
            width = self._content_width
        else:
            width = self._width
        return self.left + width

    @property
    def bottom(self) -> float:
        """
        The y-coordinate of the bottom side of the layout.

        :type: int
        """
        return self._y + self._anchor_bottom

    @property
    def top(self) -> float:
        """
        The y-coordinate of the top side of the layout.

        :type: int
        """
        if self._height is None:
            height = self._content_height
        else:
            height = self._height
        return self.bottom + height

    def _wrap_lines_invariant(self) -> None:
        self._wrap_lines = self._multiline and self._wrap_lines_flag
        assert not self._wrap_lines or self._width, "When the parameters 'multiline' and 'wrap_lines' are True, the parameter 'width' must be a number."

    def parse_distance(self, distance: Optional[Union[str, int, float]]) -> Optional[int]:
        if distance is None:
            return None
        return _parse_distance(distance, self._dpi)

    def begin_update(self) -> None:
        """Indicate that a number of changes to the layout or document
        are about to occur.

        Changes to the layout or document between calls to `begin_update` and
        `end_update` do not trigger any costly relayout of text.  Relayout of
        all changes is performed when `end_update` is called.

        Note that between the `begin_update` and `end_update` calls, values
        such as `content_width` and `content_height` are undefined (i.e., they
        may or may not be updated to reflect the latest changes).
        """
        self._update_enabled = False

    def end_update(self) -> None:
        """Perform pending layout changes since `begin_update`.

        See `begin_update`.
        """
        self._update_enabled = True
        self._update()

    def delete(self) -> None:
        """Remove this layout from its batch.
        """
        for box in self._boxes:
            box.delete(self)
        self._vertex_lists.clear()
        self._boxes.clear()

    def get_as_texture(self, min_filter=GL_NEAREST, mag_filter=GL_NEAREST) -> Texture:
        """Returns a Texture with the TextLayout drawn to it. Each call to this function returns a new
        Texture.
        ~Warning: Usage is recommended only if you understand how texture generation affects your application.
        """
        framebuffer = pyglet.image.Framebuffer()
        temp_pos = self.position
        width = int(round(self._content_width))
        height = int(round(self._content_height))
        texture = pyglet.image.Texture.create(width, height, min_filter=min_filter, mag_filter=mag_filter)
        depth_buffer = pyglet.image.buffer.Renderbuffer(width, height, GL_DEPTH_COMPONENT)
        framebuffer.attach_texture(texture)
        framebuffer.attach_renderbuffer(depth_buffer, attachment=GL_DEPTH_ATTACHMENT)
        self.position = (0 - self._anchor_left, 0 - self._anchor_bottom, 0)
        framebuffer.bind()
        self.draw()
        framebuffer.unbind()
        self.position = temp_pos
        return texture

    def draw(self) -> None:
        """Draw this text layout.

        Note that this method performs very badly if a batch was supplied to
        the constructor.  If you add this layout to a batch, you should
        ideally use only the batch's draw method.

        If this is not its own batch, InlineElements will not be drawn.
        """
        if self._own_batch:
            self._batch.draw()
        else:
            self._batch.draw_subset(self._vertex_lists)

    def _get_lines(self) -> List[_Line]:
        len_text = len(self._document.text)
        glyphs = self._get_glyphs()
        owner_runs = runlist.RunList(len_text, None)
        self._get_owner_runs(owner_runs, glyphs, 0, len_text)
        lines = [line for line in self._flow_glyphs(glyphs, owner_runs, 0, len_text)]
        self._content_width = 0
        self._line_count = len(lines)
        self._flow_lines(lines, 0, self._line_count)
        return lines

    def _update(self) -> None:
        if not self._update_enabled:
            return
        for box in self._boxes:
            box.delete(self)
        self._vertex_lists.clear()
        self._boxes.clear()
        self.group_cache.clear()
        if not self._document or not self._document.text:
            self._ascent = 0
            self._descent = 0
            self._anchor_left = 0
            self._anchor_bottom = 0
            return
        lines = self._get_lines()
        self._ascent = lines[0].ascent
        self._descent = lines[0].descent
        colors_iter = self._document.get_style_runs('color')
        background_iter = self._document.get_style_runs('background_color')
        self._anchor_left = self._get_left_anchor()
        self._anchor_bottom = self._get_bottom_anchor()
        anchor_top = self._get_top_anchor()
        context = _StaticLayoutContext(self, self._document, colors_iter, background_iter)
        for line in lines:
            self._boxes.extend(line.boxes)
            self._create_vertex_lists(line.x, line.y, self._anchor_left, anchor_top, line.start, line.boxes, context)

    def _update_color(self) -> None:
        colors_iter = self._document.get_style_runs('color')
        colors = []
        for start, end, color in colors_iter.ranges(0, colors_iter.end):
            if color is None:
                color = (0, 0, 0, 255)
            colors.extend(color * (end - start))
        start = 0
        for box in self._boxes:
            box.update_colors(colors[start:start + box.length * 4])
            start += box.length * 4

    def _get_left_anchor(self) -> int:
        """Returns the anchor for the X axis from the left."""
        if self._multiline:
            width = self._width if self._wrap_lines else self._content_width
        else:
            width = self._content_width
        if self._anchor_x == 'left':
            return 0
        elif self._anchor_x == 'center':
            return -(width // 2)
        elif self._anchor_x == 'right':
            return -width
        else:
            assert False, '`anchor_x` must be either "left", "center", or "right".'

    def _get_top_anchor(self) -> float:
        """Returns the anchor for the Y axis from the top."""
        if self._height is None:
            height = self._content_height
            offset = 0
        else:
            height = self._height
            if self._content_valign == 'top':
                offset = 0
            elif self._content_valign == 'bottom':
                offset = max(0, self._height - self._content_height)
            elif self._content_valign == 'center':
                offset = max(0, self._height - self._content_height) // 2
            else:
                assert False, '`content_valign` must be either "top", "bottom", or "center".'
        if self._anchor_y == 'top':
            return -offset
        elif self._anchor_y == 'baseline':
            return self._ascent - offset
        elif self._anchor_y == 'bottom':
            return height - offset
        elif self._anchor_y == 'center':
            if self._line_count == 1 and self._height is None:
                return self._ascent // 2 - self._descent // 4
            else:
                return height // 2 - offset
        else:
            assert False, '`anchor_y` must be either "top", "bottom", "center", or "baseline".'

    def _get_bottom_anchor(self) -> float:
        """Returns the anchor for the Y axis from the bottom."""
        if self._height is None:
            height = self._content_height
            offset = 0
        else:
            height = self._height
            if self._content_valign == 'top':
                offset = min(0, self._height - self._content_height)
            elif self._content_valign == 'bottom':
                offset = 0
            elif self._content_valign == 'center':
                offset = min(0, self._height - self._content_height) // 2
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

    def _init_document(self) -> None:
        self._update()

    def _uninit_document(self) -> None:
        pass

    def on_insert_text(self, start: int, text: str) -> None:
        """Event handler for `AbstractDocument.on_insert_text`.

        The event handler is bound by the text layout; there is no need for
        applications to interact with this method.
        """
        self._init_document()

    def on_delete_text(self, start: int, end: int) -> None:
        """Event handler for `AbstractDocument.on_delete_text`.

        The event handler is bound by the text layout; there is no need for
        applications to interact with this method.
        """
        self._init_document()

    def on_style_text(self, start: int, end: int, attributes: dict[str, Any]) -> None:
        """Event handler for `AbstractDocument.on_style_text`.

        The event handler is bound by the text layout; there is no need for
        applications to interact with this method.
        """
        if len(attributes) == 1 and 'color' in attributes.keys():
            self._update_color()
        else:
            self._init_document()

    def _get_glyphs(self) -> List[Union[_InlineElementBox, Glyph]]:
        glyphs = []
        runs = runlist.ZipRunIterator((self._document.get_font_runs(dpi=self._dpi), self._document.get_element_runs()))
        text = self._document.text
        for start, end, (font, element) in runs.ranges(0, len(text)):
            if element:
                glyphs.append(_InlineElementBox(element))
            else:
                glyphs.extend(font.get_glyphs(text[start:end]))
        return glyphs

    def _get_owner_runs(self, owner_runs: runlist.RunList, glyphs: List[Union[_InlineElementBox, Glyph]], start: int, end: int) -> None:
        owner = glyphs[start].owner
        run_start = start
        for i, glyph in enumerate(glyphs[start:end]):
            if owner != glyph.owner:
                owner_runs.set_run(run_start, i + start, owner)
                owner = glyph.owner
                run_start = i + start
        owner_runs.set_run(run_start, end, owner)

    def _flow_glyphs_wrap(self, glyphs: List[Union[_InlineElementBox, Glyph]], owner_runs: runlist.RunList, start: int, end: int) -> Iterator[_Line]:
        """Word-wrap styled text into lines of fixed width.

        Fits `glyphs` in range `start` to `end` into `_Line` s which are
        then yielded.
        """
        owner_iterator = owner_runs.get_run_iterator().ranges(start, end)
        font_iterator = self._document.get_font_runs(dpi=self._dpi)
        align_iterator = runlist.FilteredRunIterator(self._document.get_style_runs('align'), lambda value: value in ('left', 'right', 'center'), 'left')
        if self._width is None:
            wrap_iterator = runlist.ConstRunIterator(len(self.document.text), False)
        else:
            wrap_iterator = runlist.FilteredRunIterator(self._document.get_style_runs('wrap'), lambda value: value in (True, False, 'char', 'word'), True)
        margin_left_iterator = runlist.FilteredRunIterator(self._document.get_style_runs('margin_left'), lambda value: value is not None, 0)
        margin_right_iterator = runlist.FilteredRunIterator(self._document.get_style_runs('margin_right'), lambda value: value is not None, 0)
        indent_iterator = runlist.FilteredRunIterator(self._document.get_style_runs('indent'), lambda value: value is not None, 0)
        kerning_iterator = runlist.FilteredRunIterator(self._document.get_style_runs('kerning'), lambda value: value is not None, 0)
        tab_stops_iterator = runlist.FilteredRunIterator(self._document.get_style_runs('tab_stops'), lambda value: value is not None, [])
        line = _Line(start)
        line.align = align_iterator[start]
        line.margin_left = self.parse_distance(margin_left_iterator[start])
        line.margin_right = self.parse_distance(margin_right_iterator[start])
        if start == 0 or self.document.text[start - 1] in u'\n\u2029':
            line.paragraph_begin = True
            line.margin_left += self.parse_distance(indent_iterator[start])
        wrap = wrap_iterator[start]
        if self._wrap_lines:
            width = self._width - line.margin_left - line.margin_right
        x = 0
        run_accum = []
        run_accum_width = 0
        eol_ws = 0
        font = None
        for start, end, owner in owner_iterator:
            font = font_iterator[start]
            owner_accum = []
            owner_accum_width = 0
            owner_accum_commit = []
            owner_accum_commit_width = 0
            nokern = True
            index = start
            for text, glyph in zip(self.document.text[start:end], glyphs[start:end]):
                if nokern:
                    kern = 0
                    nokern = False
                else:
                    kern = self.parse_distance(kerning_iterator[index])
                if wrap != 'char' and text in u' \u200b\t':
                    for run in run_accum:
                        line.add_box(run)
                    run_accum = []
                    run_accum_width = 0
                    if text == '\t':
                        for tab_stop in tab_stops_iterator[index]:
                            tab_stop = self.parse_distance(tab_stop)
                            if tab_stop > x + line.margin_left:
                                break
                        else:
                            tab = 50.0
                            tab_stop = ((x + line.margin_left) // tab + 1) * tab
                        kern = int(tab_stop - x - line.margin_left - glyph.advance)
                    owner_accum.append((kern, glyph))
                    owner_accum_commit.extend(owner_accum)
                    owner_accum_commit_width += owner_accum_width + glyph.advance + kern
                    eol_ws += glyph.advance + kern
                    owner_accum = []
                    owner_accum_width = 0
                    x += glyph.advance + kern
                    index += 1
                    next_start = index
                else:
                    new_paragraph = text in u'\n\u2029'
                    new_line = text == u'\u2028' or new_paragraph
                    if wrap and self._wrap_lines and (x + kern + glyph.advance >= width) or new_line:
                        if new_line or wrap == 'char':
                            for run in run_accum:
                                line.add_box(run)
                            run_accum = []
                            run_accum_width = 0
                            owner_accum_commit.extend(owner_accum)
                            owner_accum_commit_width += owner_accum_width
                            owner_accum = []
                            owner_accum_width = 0
                            line.length += 1
                            next_start = index
                            if new_line:
                                next_start += 1
                        if owner_accum_commit:
                            line.add_box(_GlyphBox(owner, font, owner_accum_commit, owner_accum_commit_width))
                            owner_accum_commit = []
                            owner_accum_commit_width = 0
                        if new_line and (not line.boxes):
                            line.ascent = font.ascent
                            line.descent = font.descent
                        if line.boxes or new_line:
                            line.width -= eol_ws
                            if new_paragraph:
                                line.paragraph_end = True
                            yield line
                            try:
                                line = _Line(next_start)
                                line.align = align_iterator[next_start]
                                line.margin_left = self.parse_distance(margin_left_iterator[next_start])
                                line.margin_right = self.parse_distance(margin_right_iterator[next_start])
                            except IndexError:
                                return
                            if new_paragraph:
                                line.paragraph_begin = True
                            if run_accum and hasattr(run_accum, 'glyphs') and run_accum.glyphs:
                                k, g = run_accum[0].glyphs[0]
                                run_accum[0].glyphs[0] = (0, g)
                                run_accum_width -= k
                            elif owner_accum:
                                k, g = owner_accum[0]
                                owner_accum[0] = (0, g)
                                owner_accum_width -= k
                            else:
                                nokern = True
                            x = run_accum_width + owner_accum_width
                            if self._wrap_lines:
                                width = self._width - line.margin_left - line.margin_right
                    if isinstance(glyph, _AbstractBox):
                        run_accum.append(glyph)
                        run_accum_width += glyph.advance
                        x += glyph.advance
                    elif new_paragraph:
                        wrap = wrap_iterator[next_start]
                        line.margin_left += self.parse_distance(indent_iterator[next_start])
                        if self._wrap_lines:
                            width = self._width - line.margin_left - line.margin_right
                    elif not new_line:
                        owner_accum.append((kern, glyph))
                        owner_accum_width += glyph.advance + kern
                        x += glyph.advance + kern
                    index += 1
                    eol_ws = 0
            if owner_accum_commit:
                line.add_box(_GlyphBox(owner, font, owner_accum_commit, owner_accum_commit_width))
            if owner_accum:
                run_accum.append(_GlyphBox(owner, font, owner_accum, owner_accum_width))
                run_accum_width += owner_accum_width
        for run in run_accum:
            line.add_box(run)
        if not line.boxes:
            if font is None:
                font = self._document.get_font(0, dpi=self._dpi)
            line.ascent = font.ascent
            line.descent = font.descent
        yield line

    def _flow_glyphs_single_line(self, glyphs: List[Union[_InlineElementBox, Glyph]], owner_runs: runlist.RunList, start: int, end: int) -> Iterator[_Line]:
        owner_iterator = owner_runs.get_run_iterator().ranges(start, end)
        font_iterator = self.document.get_font_runs(dpi=self._dpi)
        kern_iterator = runlist.FilteredRunIterator(self.document.get_style_runs('kerning'), lambda value: value is not None, 0)
        line = _Line(start)
        font = font_iterator[0]
        if self._width:
            align_iterator = runlist.FilteredRunIterator(self._document.get_style_runs('align'), lambda value: value in ('left', 'right', 'center'), 'left')
            line.align = align_iterator[start]
        for start, end, owner in owner_iterator:
            font = font_iterator[start]
            width = 0
            owner_glyphs = []
            for kern_start, kern_end, kern in kern_iterator.ranges(start, end):
                gs = glyphs[kern_start:kern_end]
                width += sum([g.advance for g in gs])
                width += kern * (kern_end - kern_start)
                owner_glyphs.extend(zip([kern] * (kern_end - kern_start), gs))
            if owner is None:
                for kern, glyph in owner_glyphs:
                    line.add_box(glyph)
            else:
                line.add_box(_GlyphBox(owner, font, owner_glyphs, width))
        if not line.boxes:
            line.ascent = font.ascent
            line.descent = font.descent
        line.paragraph_begin = line.paragraph_end = True
        yield line

    def _flow_lines(self, lines: List[_Line], start: int, end: int) -> int:
        margin_top_iterator = runlist.FilteredRunIterator(self._document.get_style_runs('margin_top'), lambda value: value is not None, 0)
        margin_bottom_iterator = runlist.FilteredRunIterator(self._document.get_style_runs('margin_bottom'), lambda value: value is not None, 0)
        line_spacing_iterator = self._document.get_style_runs('line_spacing')
        leading_iterator = runlist.FilteredRunIterator(self._document.get_style_runs('leading'), lambda value: value is not None, 0)
        if start == 0:
            y = 0
        else:
            line = lines[start - 1]
            line_spacing = self.parse_distance(line_spacing_iterator[line.start])
            leading = self.parse_distance(leading_iterator[line.start])
            y = line.y
            if line_spacing is None:
                y += line.descent
            if line.paragraph_end:
                y -= self.parse_distance(margin_bottom_iterator[line.start])
        line_index = start
        for line in lines[start:]:
            if line.paragraph_begin:
                y -= self.parse_distance(margin_top_iterator[line.start])
                line_spacing = self.parse_distance(line_spacing_iterator[line.start])
                leading = self.parse_distance(leading_iterator[line.start])
            else:
                y -= leading
            if line_spacing is None:
                y -= line.ascent
            else:
                y -= line_spacing
            if line.align == 'left' or line.width > self.width:
                line.x = line.margin_left
            elif line.align == 'center':
                line.x = (self.width - line.margin_left - line.margin_right - line.width) // 2 + line.margin_left
            elif line.align == 'right':
                line.x = self.width - line.margin_right - line.width
            self._content_width = max(self._content_width, line.width + line.margin_left)
            if line.y == y and line_index >= end:
                break
            line.y = y
            if line_spacing is None:
                y += line.descent
            if line.paragraph_end:
                y -= self.parse_distance(margin_bottom_iterator[line.start])
            line_index += 1
        else:
            self._content_height = -y
        return line_index

    def _create_vertex_lists(self, line_x: float, line_y: float, anchor_x: float, anchor_y: float, i: int, boxes: List[_AbstractBox], context: _LayoutContext):
        acc_anchor_x = anchor_x
        for box in boxes:
            box.place(self, i, self._x, self._y, self._z, line_x, line_y, self._rotation, self._visible, acc_anchor_x, anchor_y, context)
            i += box.length
            acc_anchor_x += box.advance