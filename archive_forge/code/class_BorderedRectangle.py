import math
from abc import ABC, abstractmethod
import pyglet
from pyglet.gl import GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_BLEND, GL_TRIANGLES
from pyglet.gl import glBlendFunc, glEnable, glDisable
from pyglet.graphics import Batch, Group
from pyglet.math import Vec2
class BorderedRectangle(ShapeBase):

    def __init__(self, x, y, width, height, border=1, color=(255, 255, 255), border_color=(100, 100, 100), batch=None, group=None):
        """Create a rectangle or square.

        The rectangle's anchor point defaults to the (x, y) coordinates,
        which are at the bottom left.

        :Parameters:
            `x` : float
                The X coordinate of the rectangle.
            `y` : float
                The Y coordinate of the rectangle.
            `width` : float
                The width of the rectangle.
            `height` : float
                The height of the rectangle.
            `border` : float
                The thickness of the border.
            `color` : (int, int, int, int)
                The RGB or RGBA fill color of the rectangle, specified
                as a tuple of 3 or 4 ints in the range of 0-255. RGB
                colors will be treated as having an opacity of 255.
            `border_color` : (int, int, int, int)
                The RGB or RGBA fill color of the border, specified
                as a tuple of 3 or 4 ints in the range of 0-255. RGB
                colors will be treated as having an opacity of 255.

                The alpha values must match if you pass RGBA values to
                both this argument and `border_color`. If they do not,
                a `ValueError` will be raised informing you of the
                ambiguity.
            `batch` : `~pyglet.graphics.Batch`
                Optional batch to add the rectangle to.
            `group` : `~pyglet.graphics.Group`
                Optional parent group of the rectangle.
        """
        self._x = x
        self._y = y
        self._width = width
        self._height = height
        self._rotation = 0
        self._border = border
        self._num_verts = 8
        fill_r, fill_g, fill_b, *fill_a = color
        border_r, border_g, border_b, *border_a = border_color
        alpha = 255
        if fill_a and border_a and (fill_a[0] != border_a[0]):
            raise ValueError('When color and border_color are both RGBA values,they must both have the same opacity')
        elif fill_a:
            alpha = fill_a[0]
        elif border_a:
            alpha = border_a[0]
        self._rgba = (fill_r, fill_g, fill_b, alpha)
        self._border_rgba = (border_r, border_g, border_b, alpha)
        program = get_default_shader()
        self._batch = batch or Batch()
        self._group = self.group_class(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, program, group)
        self._create_vertex_list()

    def __contains__(self, point):
        assert len(point) == 2
        point = _rotate_point((self._x, self._y), point, math.radians(self._rotation))
        x, y = (self._x - self._anchor_x, self._y - self._anchor_y)
        return x < point[0] < x + self._width and y < point[1] < y + self._height

    def _create_vertex_list(self):
        indices = [0, 1, 2, 0, 2, 3, 0, 4, 3, 4, 7, 3, 0, 1, 5, 0, 5, 4, 1, 2, 5, 5, 2, 6, 6, 2, 3, 6, 3, 7]
        self._vertex_list = self._group.program.vertex_list_indexed(8, self._draw_mode, indices, self._batch, self._group, position=('f', self._get_vertices()), colors=('Bn', self._rgba * 4 + self._border_rgba * 4), translation=('f', (self._x, self._y) * self._num_verts))

    def _update_color(self):
        self._vertex_list.colors[:] = self._rgba * 4 + self._border_rgba * 4

    def _get_vertices(self):
        if not self._visible:
            return (0, 0) * self._num_verts
        else:
            bx1 = -self._anchor_x
            by1 = -self._anchor_y
            bx2 = bx1 + self._width
            by2 = by1 + self._height
            b = self._border
            ix1 = bx1 + b
            iy1 = by1 + b
            ix2 = bx2 - b
            iy2 = by2 - b
            return (ix1, iy1, ix2, iy1, ix2, iy2, ix1, iy2, bx1, by1, bx2, by1, bx2, by2, bx1, by2)

    def _update_vertices(self):
        self._vertex_list.position[:] = self._get_vertices()

    @property
    def border(self):
        """The border width of the rectangle.

        :return: float
        """
        return self._border

    @border.setter
    def border(self, width):
        self._border = width
        self._update_vertices()

    @property
    def width(self):
        """The width of the rectangle.

        :type: float
        """
        return self._width

    @width.setter
    def width(self, value):
        self._width = value
        self._update_vertices()

    @property
    def height(self):
        """The height of the rectangle.

        :type: float
        """
        return self._height

    @height.setter
    def height(self, value):
        self._height = value
        self._update_vertices()

    @property
    def border_color(self):
        """The rectangle's border color.

        This property sets the color of the border of a bordered rectangle.

        The color is specified as an RGB tuple of integers '(red, green, blue)'
        or an RGBA tuple of integers '(red, green, blue, alpha)`. Setting the
        alpha on this property will change the alpha of the entire shape,
        including both the fill and the border.

        Each color component must be in the range 0 (dark) to 255 (saturated).

        :type: (int, int, int, int)
        """
        return self._border_rgba

    @border_color.setter
    def border_color(self, values):
        r, g, b, *a = values
        if a:
            alpha = a[0]
        else:
            alpha = self._rgba[3]
        self._border_rgba = (r, g, b, alpha)
        self._rgba = (*self._rgba[:3], alpha)
        self._update_color()

    @property
    def color(self):
        """The rectangle's fill color.

        This property sets the color of the inside of a bordered rectangle.

        The color is specified as an RGB tuple of integers '(red, green, blue)'
        or an RGBA tuple of integers '(red, green, blue, alpha)`. Setting the
        alpha on this property will change the alpha of the entire shape,
        including both the fill and the border.

        Each color component must be in the range 0 (dark) to 255 (saturated).

        :type: (int, int, int, int)
        """
        return self._rgba

    @color.setter
    def color(self, values):
        r, g, b, *a = values
        if a:
            alpha = a[0]
        else:
            alpha = self._rgba[3]
        self._rgba = (r, g, b, alpha)
        self._border_rgba = (*self._border_rgba[:3], alpha)
        self._update_color()