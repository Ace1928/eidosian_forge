from typing import TYPE_CHECKING, Tuple, Optional
import pyglet
class TextureBin:
    """Collection of texture atlases.

    :py:class:`~pyglet.image.atlas.TextureBin` maintains a collection of texture atlases, and creates new
    ones as necessary to accommodate images added to the bin.
    """

    def __init__(self, texture_width: int=2048, texture_height: int=2048) -> None:
        """Create a texture bin for holding atlases of the given size.

        :Parameters:
            `texture_width` : int
                Width of texture atlases to create.
            `texture_height` : int
                Height of texture atlases to create.
            `border` : int
                Leaves specified pixels of blank space around
                each image added to the Atlases.

        """
        max_texture_size = pyglet.image.get_max_texture_size()
        self.texture_width = min(texture_width, max_texture_size)
        self.texture_height = min(texture_height, max_texture_size)
        self.atlases = []

    def add(self, img: 'AbstractImage', border: int=0) -> 'TextureRegion':
        """Add an image into this texture bin.

        This method calls `TextureAtlas.add` for the first atlas that has room
        for the image.

        `AllocatorException` is raised if the image exceeds the dimensions of
        ``texture_width`` and ``texture_height``.

        :Parameters:
            `img` : `~pyglet.image.AbstractImage`
                The image to add.
            `border` : int
                Leaves specified pixels of blank space around
                each image added to the Atlas.

        :rtype: :py:class:`~pyglet.image.TextureRegion`
        :return: The region of an atlas containing the newly added image.
        """
        for atlas in list(self.atlases):
            try:
                return atlas.add(img, border)
            except AllocatorException:
                if img.width < 64 and img.height < 64:
                    self.atlases.remove(atlas)
        atlas = TextureAtlas(self.texture_width, self.texture_height)
        self.atlases.append(atlas)
        return atlas.add(img, border)