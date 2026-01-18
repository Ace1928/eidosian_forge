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
class OSM(GoogleWTS):

    def _image_url(self, tile):
        x, y, z = tile
        return f'https://a.tile.openstreetmap.org/{z}/{x}/{y}.png'