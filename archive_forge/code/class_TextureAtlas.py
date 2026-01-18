from typing import TYPE_CHECKING, Tuple, Optional
import pyglet
class TextureAtlas:
    """Collection of images within a texture."""

    def __init__(self, width: int=2048, height: int=2048) -> None:
        """Create a texture atlas of the given size.

        :Parameters:
            `width` : int
                Width of the underlying texture.
            `height` : int
                Height of the underlying texture.

        """
        max_texture_size = pyglet.image.get_max_texture_size()
        width = min(width, max_texture_size)
        height = min(height, max_texture_size)
        self.texture = pyglet.image.Texture.create(width, height)
        self.allocator = Allocator(width, height)

    def add(self, img: 'AbstractImage', border: int=0) -> 'TextureRegion':
        """Add an image to the atlas.

        This method will fail if the given image cannot be transferred
        directly to a texture (for example, if it is another texture).
        :py:class:`~pyglet.image.ImageData` is the usual image type for this method.

        `AllocatorException` will be raised if there is no room in the atlas
        for the image.

        :Parameters:
            `img` : `~pyglet.image.AbstractImage`
                The image to add.
            `border` : int
                Leaves specified pixels of blank space around
                each image added to the Atlas.

        :rtype: :py:class:`~pyglet.image.TextureRegion`
        :return: The region of the atlas containing the newly added image.
        """
        x, y = self.allocator.alloc(img.width + border * 2, img.height + border * 2)
        self.texture.blit_into(img, x + border, y + border, 0)
        return self.texture.get_region(x + border, y + border, img.width, img.height)