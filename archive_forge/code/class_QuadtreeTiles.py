from abc import ABCMeta, abstractmethod
import concurrent.futures
import io
from pathlib import Path
import warnings
import numpy as np
from PIL import Image
import shapely.geometry as sgeom
import cartopy
import cartopy.crs as ccrs
class QuadtreeTiles(GoogleWTS):
    """
    Implement web tile retrieval using the Microsoft WTS quadkey coordinate
    system.

    A "tile" in this class refers to a quadkey such as "1", "14" or "141"
    where the length of the quatree is the zoom level in Google Tile terms.

    """

    def _image_url(self, tile):
        return f'http://ecn.dynamic.t1.tiles.virtualearth.net/comp/CompositionHandler/{tile}?mkt=en-gb&it=A,G,L&shading=hill&n=z'

    def tms_to_quadkey(self, tms, google=False):
        quadKey = ''
        x, y, z = tms
        if not google:
            y = 2 ** z - 1 - y
        for i in range(z, 0, -1):
            digit = 0
            mask = 1 << i - 1
            if x & mask != 0:
                digit += 1
            if y & mask != 0:
                digit += 2
            quadKey += str(digit)
        return quadKey

    def quadkey_to_tms(self, quadkey, google=False):
        assert isinstance(quadkey, str), 'quadkey must be a string'
        x = y = 0
        z = len(quadkey)
        for i in range(z, 0, -1):
            mask = 1 << i - 1
            if quadkey[z - i] == '0':
                pass
            elif quadkey[z - i] == '1':
                x |= mask
            elif quadkey[z - i] == '2':
                y |= mask
            elif quadkey[z - i] == '3':
                x |= mask
                y |= mask
            else:
                raise ValueError(f'Invalid QuadKey digit sequence: {quadkey}')
        if not google:
            y = 2 ** z - 1 - y
        return (x, y, z)

    def subtiles(self, quadkey):
        for i in range(4):
            yield (quadkey + str(i))

    def tileextent(self, quadkey):
        x_y_z = self.quadkey_to_tms(quadkey, google=True)
        return GoogleWTS.tileextent(self, x_y_z)

    def find_images(self, target_domain, target_z, start_tile=None):
        """
        Find all the quadtrees at the given target zoom, in the given
        target domain.

        target_z must be a value >= 1.

        """
        if target_z == 0:
            raise ValueError('The empty quadtree cannot be returned.')
        if start_tile is None:
            start_tiles = ['0', '1', '2', '3']
        else:
            start_tiles = [start_tile]
        for start_tile in start_tiles:
            start_tile = self.quadkey_to_tms(start_tile, google=True)
            for tile in GoogleWTS.find_images(self, target_domain, target_z, start_tile=start_tile):
                yield self.tms_to_quadkey(tile, google=True)