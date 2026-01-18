from __future__ import annotations
from io import BytesIO
import math
import os
import dask
import dask.bag as db
import numpy as np
from PIL.Image import fromarray
class FileSystemTileRenderer(TileRenderer):

    def render(self, da, level):
        for img, x, y, z in super(FileSystemTileRenderer, self).render(da, level):
            tile_file_name = '{}.{}'.format(y, self.tile_format.lower())
            tile_directory = os.path.join(self.output_location, str(z), str(x))
            output_file = os.path.join(tile_directory, tile_file_name)
            _create_dir(tile_directory)
            img.save(output_file, self.tile_format)