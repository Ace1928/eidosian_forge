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
class AzureMapsTiles(GoogleWTS):

    def __init__(self, subscription_key, tileset_id='microsoft.imagery', api_version='2.0', desired_tile_form='RGB', cache=False):
        """
        Set up a new instance to retrieve tiles from Azure Maps.

        Access to Azure Maps REST API requires a subscription key.
        See https://docs.microsoft.com/en-us/azure/azure-maps/azure-maps-authentication#shared-key-authentication/  # noqa: E501
        for details.

        Parameters
        ----------
        subscription_key
            A valid Azure Maps subscription key.
        tileset_id
            A tileset ID for a map. See
            https://docs.microsoft.com/en-us/rest/api/maps/renderv2/getmaptilepreview#tilesetid
            for details.
        api_version
            API version to use. Defaults to 2.0 as recommended by Microsoft.

        """
        super().__init__(desired_tile_form=desired_tile_form, cache=cache)
        self.subscription_key = subscription_key
        self.tileset_id = tileset_id
        self.api_version = api_version

    def _image_url(self, tile):
        x, y, z = tile
        return f'https://atlas.microsoft.com/map/tile?api-version={self.api_version}&tilesetId={self.tileset_id}&x={x}&y={y}&zoom={z}&subscription-key={self.subscription_key}'