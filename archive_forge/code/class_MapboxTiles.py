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
class MapboxTiles(GoogleWTS):
    """
    Implement web tile retrieval from Mapbox.

    For terms of service, see https://www.mapbox.com/tos/.

    """

    def __init__(self, access_token, map_id, cache=False):
        """
        Set up a new Mapbox tiles instance.

        Access to Mapbox web services requires an access token and a map ID.
        See https://www.mapbox.com/api-documentation/ for details.

        Parameters
        ----------
        access_token : str
            A valid Mapbox API access token.
        map_id : str
            An ID for a publicly accessible map (provided by Mapbox).
            This is the map whose tiles will be retrieved through this process
            and is specified through the Mapbox Styles API
            (https://docs.mapbox.com/api/maps/styles/)

            Examples::

                map_id='streets-v11'
                map_id='outdoors-v11'
                map_id='satellite-v9'
        """
        self.access_token = access_token
        self.map_id = map_id
        super().__init__(cache=cache)

    def _image_url(self, tile):
        x, y, z = tile
        return f'https://api.mapbox.com/styles/v1/mapbox/{self.map_id}/tiles/{z}/{x}/{y}?access_token={self.access_token}'