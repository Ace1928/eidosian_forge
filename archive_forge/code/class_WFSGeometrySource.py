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
class WFSGeometrySource:
    """Web Feature Service (WFS) retrieval for Cartopy."""

    def __init__(self, service, features, getfeature_extra_kwargs=None):
        """
        Parameters
        ----------
        service
            The URL of a WFS, or an instance of
            :class:`owslib.wfs.WebFeatureService`.
        features
            The typename(s) of the features from the WFS that
            will be retrieved and made available as geometries.
        getfeature_extra_kwargs: optional
            Extra keyword args to pass to the service's `getfeature` call.
            Defaults to None

        """
        if WebFeatureService is None:
            raise ImportError(_OWSLIB_REQUIRED)
        if isinstance(service, str):
            self.url = urlparse(service).hostname
            service = WebFeatureService(service)
        else:
            self.url = ''
        if isinstance(features, str):
            features = [features]
        if getfeature_extra_kwargs is None:
            getfeature_extra_kwargs = {}
        if not features:
            raise ValueError('One or more features must be specified.')
        for feature in features:
            if feature not in service.contents:
                raise ValueError(f'The {feature!r} feature does not exist in this service.')
        self.service = service
        self.features = features
        self.getfeature_extra_kwargs = getfeature_extra_kwargs
        self._default_urn = None

    def default_projection(self):
        """
        Return a :class:`cartopy.crs.Projection` in which the WFS
        service can supply the requested features.

        """
        if self._default_urn is None:
            default_urn = {self.service.contents[feature].crsOptions[0] for feature in self.features}
            if len(default_urn) != 1:
                ValueError('Failed to find a single common default SRS across all features (typenames).')
            else:
                default_urn = default_urn.pop()
            if str(default_urn) not in _URN_TO_CRS and ':EPSG:' not in str(default_urn):
                raise ValueError(f'Unknown mapping from SRS/CRS_URN {default_urn!r} to cartopy projection.')
            self._default_urn = default_urn
        if str(self._default_urn) in _URN_TO_CRS:
            return _URN_TO_CRS[str(self._default_urn)]
        elif ':EPSG:' in str(self._default_urn):
            epsg_num = str(self._default_urn).split(':')[-1]
            return ccrs.epsg(int(epsg_num))
        else:
            raise ValueError(f'Unknown coordinate reference system: {str(self._default_urn)}')

    def fetch_geometries(self, projection, extent):
        """
        Return any Point, Linestring or LinearRing geometries available
        from the WFS that lie within the specified extent.

        Parameters
        ----------
        projection: :class:`cartopy.crs.Projection`
            The projection in which the extent is specified and in
            which the geometries should be returned. Only the default
            (native) projection is supported.
        extent: four element tuple
            (min_x, max_x, min_y, max_y) tuple defining the geographic extent
            of the geometries to obtain.

        Returns
        -------
        geoms
            A list of Shapely geometries.

        """
        if self.default_projection() != projection:
            raise ValueError(f'Geometries are only available in projection {self.default_projection()!r}.')
        min_x, max_x, min_y, max_y = extent
        response = self.service.getfeature(typename=self.features, bbox=(min_x, min_y, max_x, max_y), **self.getfeature_extra_kwargs)
        geoms_by_srs = self._to_shapely_geoms(response)
        if not geoms_by_srs:
            geoms = []
        elif len(geoms_by_srs) > 1:
            raise ValueError('Unexpected response from the WFS server. The geometries are in multiple SRSs, when only one was expected.')
        else:
            srs, geoms = list(geoms_by_srs.items())[0]
            if srs is not None:
                if srs in _URN_TO_CRS:
                    geom_proj = _URN_TO_CRS[srs]
                    if geom_proj != projection:
                        raise ValueError(f'The geometries are not in expected projection. Expected {projection!r}, got {geom_proj!r}.')
                elif ':EPSG:' in srs:
                    epsg_num = srs.split(':')[-1]
                    geom_proj = ccrs.epsg(int(epsg_num))
                    if geom_proj != projection:
                        raise ValueError(f'The EPSG geometries are not in expected  projection. Expected {projection!r},  got {geom_proj!r}.')
                else:
                    warnings.warn(f'Unable to verify matching projections due to incomplete mappings from SRS identifiers to cartopy projections. The geometries have an SRS of {srs!r}.')
        return geoms

    def _to_shapely_geoms(self, response):
        """
        Convert polygon coordinate strings in WFS response XML to Shapely
        geometries.

        Parameters
        ----------
        response: (file-like object)
            WFS response XML data.

        Returns
        -------
        geoms_by_srs
            A dictionary containing geometries, with key-value pairs of
            the form {srsname: [geoms]}.

        """
        linear_rings_data = []
        linestrings_data = []
        points_data = []
        tree = ElementTree.parse(response)
        for node in tree.iter():
            snode = str(node)
            if _MAP_SERVER_NS in snode or (self.url and self.url in snode):
                s1 = snode.split()[1]
                tag = s1[s1.find('}') + 1:-1]
                if 'geom' in tag or 'Geom' in tag:
                    find_str = f'.//{_GML_NS}LinearRing'
                    if self._node_has_child(node, find_str):
                        data = self._find_polygon_coords(node, find_str)
                        linear_rings_data.extend(data)
                    find_str = f'.//{_GML_NS}LineString'
                    if self._node_has_child(node, find_str):
                        data = self._find_polygon_coords(node, find_str)
                        linestrings_data.extend(data)
                    find_str = f'.//{_GML_NS}Point'
                    if self._node_has_child(node, find_str):
                        data = self._find_polygon_coords(node, find_str)
                        points_data.extend(data)
        geoms_by_srs = {}
        for srs, x, y in linear_rings_data:
            geoms_by_srs.setdefault(srs, []).append(sgeom.LinearRing(zip(x, y)))
        for srs, x, y in linestrings_data:
            geoms_by_srs.setdefault(srs, []).append(sgeom.LineString(zip(x, y)))
        for srs, x, y in points_data:
            geoms_by_srs.setdefault(srs, []).append(sgeom.Point(zip(x, y)))
        return geoms_by_srs

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

    @staticmethod
    def _node_has_child(node, find_str):
        """
        Return whether `node` contains (at any sub-level), a node with name
        equal to `find_str`.

        """
        element = node.find(find_str)
        return element is not None