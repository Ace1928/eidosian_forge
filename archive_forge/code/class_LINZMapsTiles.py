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
class LINZMapsTiles(GoogleWTS):

    def __init__(self, apikey, layer_id, api_version='v4', desired_tile_form='RGB', cache=False):
        """
        Set up a new instance to retrieve tiles from The LINZ
        aka. Land Information New Zealand

        Access to LINZ WMTS GetCapabilities requires an API key.
        Register yourself free in https://id.koordinates.com/signup/
        to gain access into the LINZ database.

        Parameters
        ----------
        apikey
            A valid LINZ API key specific for every users.
        layer_id
            A layer ID for a map. See the "Technical Details" lower down the
            "About" tab for each layer displayed in the LINZ data service.
        api_version
            API version to use. Defaults to v4 for now.

        """
        super().__init__(desired_tile_form=desired_tile_form, cache=cache)
        self.apikey = apikey
        self.layer_id = layer_id
        self.api_version = api_version

    def _image_url(self, tile):
        x, y, z = tile
        return f'https://tiles-a.koordinates.com/services;key={self.apikey}/tiles/{self.api_version}/layer={self.layer_id}/EPSG:3857/{z}/{x}/{y}.png'