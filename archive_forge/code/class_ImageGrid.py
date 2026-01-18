import re
import weakref
from ctypes import *
from io import open, BytesIO
import pyglet
from pyglet.gl import *
from pyglet.gl import gl_info
from pyglet.util import asbytes
from .codecs import ImageEncodeException, ImageDecodeException
from .codecs import registry as _codec_registry
from .codecs import add_default_codecs as _add_default_codecs
from .animation import Animation, AnimationFrame
from .buffer import *
from . import atlas
class ImageGrid(AbstractImage, AbstractImageSequence):
    """An imaginary grid placed over an image allowing easy access to
    regular regions of that image.

    The grid can be accessed either as a complete image, or as a sequence
    of images.  The most useful applications are to access the grid
    as a :py:class:`~pyglet.image.TextureGrid`::

        image_grid = ImageGrid(...)
        texture_grid = image_grid.get_texture_sequence()

    or as a :py:class:`~pyglet.image.Texture3D`::

        image_grid = ImageGrid(...)
        texture_3d = Texture3D.create_for_image_grid(image_grid)

    """
    _items = ()
    _texture_grid = None

    def __init__(self, image, rows, columns, item_width=None, item_height=None, row_padding=0, column_padding=0):
        """Construct a grid for the given image.

        You can specify parameters for the grid, for example setting
        the padding between cells.  Grids are always aligned to the
        bottom-left corner of the image.

        :Parameters:
            `image` : AbstractImage
                Image over which to construct the grid.
            `rows` : int
                Number of rows in the grid.
            `columns` : int
                Number of columns in the grid.
            `item_width` : int
                Width of each column.  If unspecified, is calculated such
                that the entire image width is used.
            `item_height` : int
                Height of each row.  If unspecified, is calculated such that
                the entire image height is used.
            `row_padding` : int
                Pixels separating adjacent rows.  The padding is only
                inserted between rows, not at the edges of the grid.
            `column_padding` : int
                Pixels separating adjacent columns.  The padding is only
                inserted between columns, not at the edges of the grid.
        """
        super().__init__(image.width, image.height)
        self.image = image
        self.rows = rows
        self.columns = columns
        self.item_width = item_width or (image.width - column_padding * (columns - 1)) // columns
        self.item_height = item_height or (image.height - row_padding * (rows - 1)) // rows
        self.row_padding = row_padding
        self.column_padding = column_padding

    def get_texture(self, rectangle=False):
        return self.image.get_texture(rectangle)

    def get_image_data(self):
        return self.image.get_image_data()

    def get_texture_sequence(self):
        if not self._texture_grid:
            self._texture_grid = TextureGrid(self)
        return self._texture_grid

    def __len__(self):
        return self.rows * self.columns

    def _update_items(self):
        if not self._items:
            self._items = []
            y = 0
            for row in range(self.rows):
                x = 0
                for col in range(self.columns):
                    self._items.append(self.image.get_region(x, y, self.item_width, self.item_height))
                    x += self.item_width + self.column_padding
                y += self.item_height + self.row_padding

    def __getitem__(self, index):
        self._update_items()
        if type(index) is tuple:
            row, column = index
            assert 0 <= row < self.rows and 0 <= column < self.columns
            return self._items[row * self.columns + column]
        else:
            return self._items[index]

    def __iter__(self):
        self._update_items()
        return iter(self._items)