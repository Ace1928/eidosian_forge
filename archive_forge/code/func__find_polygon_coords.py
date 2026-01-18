import collections
import io
import math
from urllib.parse import urlparse
import warnings
import weakref
from xml.etree import ElementTree
import numpy as np
from PIL import Image
import shapely.geometry as sgeom
import cartopy.crs as ccrs
from cartopy.img_transform import warp_array
from cartopy.io import LocatedImage, RasterSource
def _find_polygon_coords(self, node, find_str):
    """
        Return the x, y coordinate values for all the geometries in
        a given`node`.

        Parameters
        ----------
        node: :class:`xml.etree.ElementTree.Element`
            Node of the parsed XML response.
        find_str: string
            A search string used to match subelements that contain
            the coordinates of interest, for example:
            './/{http://www.opengis.net/gml}LineString'

        Returns
        -------
        data
            A list of (srsName, x_vals, y_vals) tuples.

        """
    data = []
    for polygon in node.findall(find_str):
        feature_srs = polygon.attrib.get('srsName')
        x, y = ([], [])
        coordinates_find_str = f'{_GML_NS}coordinates'
        coords_find_str = f'{_GML_NS}coord'
        if self._node_has_child(polygon, coordinates_find_str):
            points = polygon.findtext(coordinates_find_str)
            coords = points.strip().split(' ')
            for coord in coords:
                x_val, y_val = coord.split(',')
                x.append(float(x_val))
                y.append(float(y_val))
        elif self._node_has_child(polygon, coords_find_str):
            for coord in polygon.findall(coords_find_str):
                x.append(float(coord.findtext(f'{_GML_NS}X')))
                y.append(float(coord.findtext(f'{_GML_NS}Y')))
        else:
            raise ValueError('Unable to find or parse coordinate values from the XML.')
        data.append((feature_srs, x, y))
    return data