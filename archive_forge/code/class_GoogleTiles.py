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
class GoogleTiles(GoogleWTS):

    def __init__(self, desired_tile_form='RGB', style='street', url='https://mts0.google.com/vt/lyrs={style}@177000000&hl=en&src=api&x={x}&y={y}&z={z}&s=G', cache=False):
        """
        Parameters
        ----------
        desired_tile_form: optional
            Defaults to 'RGB'.
        style: optional
            The style for the Google Maps tiles.  One of 'street',
            'satellite', 'terrain', and 'only_streets'.  Defaults to 'street'.
        url: optional
            URL pointing to a tile source and containing {x}, {y}, and {z}.
            Such as: ``'https://server.arcgisonline.com/ArcGIS/rest/services/World_Shaded_Relief/MapServer/tile/{z}/{y}/{x}.jpg'``

        """
        styles = ['street', 'satellite', 'terrain', 'only_streets']
        style = style.lower()
        self.url = url
        if style not in styles:
            raise ValueError(f'Invalid style {style!r}. Valid styles: {', '.join(styles)}')
        self.style = style
        if self.style in ['satellite', 'terrain'] and (not hasattr(Image.core, 'jpeg_decoder')) or not Image.core.jpeg_decoder:
            raise ValueError(f'The {self.style!r} style requires pillow with jpeg decoding support.')
        return super().__init__(desired_tile_form=desired_tile_form, cache=cache)

    def _image_url(self, tile):
        style_dict = {'street': 'm', 'satellite': 's', 'terrain': 't', 'only_streets': 'h'}
        url = self.url.format(style=style_dict[self.style], x=tile[0], X=tile[0], y=tile[1], Y=tile[1], z=tile[2], Z=tile[2])
        return url