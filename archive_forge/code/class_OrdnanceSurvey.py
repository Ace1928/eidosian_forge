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
class OrdnanceSurvey(GoogleWTS):
    """
    Implement web tile retrieval from Ordnance Survey map data.
    To use this tile image source you will need to obtain an
    API key from Ordnance Survey. You can get a free API key from
    https://osdatahub.os.uk

    For more details on Ordnance Survey layer styles, see
    https://osdatahub.os.uk/docs/wmts/technicalSpecification.

    For the API framework agreement, see
    https://osdatahub.os.uk/legal/apiTermsConditions.
    """

    def __init__(self, apikey, layer='Road_3857', desired_tile_form='RGB', cache=False):
        """
        Parameters
        ----------
        apikey: required
            The authentication key provided by OS to query the maps API
        layer: optional
            The style of the Ordnance Survey map tiles. One of 'Outdoor',
            'Road', 'Light', 'Night', 'Leisure'. Defaults to 'Road'.
            Details about the style of layer can be found at:

            - https://apidocs.os.uk/docs/layer-information
            - https://apidocs.os.uk/docs/map-styles
        desired_tile_form: optional
            Defaults to 'RGB'.
        """
        super().__init__(desired_tile_form=desired_tile_form, cache=cache)
        self.apikey = apikey
        if layer not in ('Road_3857', 'Outdoor_3857', 'Light_3857', 'Road', 'Outdoor', 'Light'):
            raise ValueError(f'Invalid layer {layer}')
        elif layer in ('Road', 'Outdoor', 'Light'):
            layer += '_3857'
        self.layer = layer

    def _image_url(self, tile):
        x, y, z = tile
        return f'https://api.os.uk/maps/raster/v1/zxy/{self.layer}/{z}/{x}/{y}.png?key={self.apikey}'