from binascii import b2a_base64, hexlify
import html
import json
import mimetypes
import os
import struct
import warnings
from copy import deepcopy
from os.path import splitext
from pathlib import Path, PurePath
from IPython.utils.py3compat import cast_unicode
from IPython.testing.skipdoctest import skip_doctest
from . import display_functions
from warnings import warn
class GeoJSON(JSON):
    """GeoJSON expects JSON-able dict

    not an already-serialized JSON string.

    Scalar types (None, number, string) are not allowed, only dict containers.
    """

    def __init__(self, *args, **kwargs):
        """Create a GeoJSON display object given raw data.

        Parameters
        ----------
        data : dict or list
            VegaLite data. Not an already-serialized JSON string.
            Scalar types (None, number, string) are not allowed, only dict
            or list containers.
        url_template : string
            Leaflet TileLayer URL template: http://leafletjs.com/reference.html#url-template
        layer_options : dict
            Leaflet TileLayer options: http://leafletjs.com/reference.html#tilelayer-options
        url : unicode
            A URL to download the data from.
        filename : unicode
            Path to a local file to load the data from.
        metadata : dict
            Specify extra metadata to attach to the json display object.

        Examples
        --------
        The following will display an interactive map of Mars with a point of
        interest on frontend that do support GeoJSON display.

            >>> from IPython.display import GeoJSON

            >>> GeoJSON(data={
            ...     "type": "Feature",
            ...     "geometry": {
            ...         "type": "Point",
            ...         "coordinates": [-81.327, 296.038]
            ...     }
            ... },
            ... url_template="http://s3-eu-west-1.amazonaws.com/whereonmars.cartodb.net/{basemap_id}/{z}/{x}/{y}.png",
            ... layer_options={
            ...     "basemap_id": "celestia_mars-shaded-16k_global",
            ...     "attribution" : "Celestia/praesepe",
            ...     "minZoom" : 0,
            ...     "maxZoom" : 18,
            ... })
            <IPython.core.display.GeoJSON object>

        In the terminal IPython, you will only see the text representation of
        the GeoJSON object.

        """
        super(GeoJSON, self).__init__(*args, **kwargs)

    def _ipython_display_(self):
        bundle = {'application/geo+json': self.data, 'text/plain': '<IPython.display.GeoJSON object>'}
        metadata = {'application/geo+json': self.metadata}
        display_functions.display(bundle, metadata=metadata, raw=True)