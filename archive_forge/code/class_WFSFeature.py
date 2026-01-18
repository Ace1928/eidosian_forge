from abc import ABCMeta, abstractmethod
import numpy as np
import shapely.geometry as sgeom
import cartopy.crs
import cartopy.io.shapereader as shapereader
class WFSFeature(Feature):
    """
    A class capable of drawing a collection of geometries
    obtained from an OGC Web Feature Service (WFS).

    This feature requires additional dependencies. If installed via pip,
    try ``pip install cartopy[ows]``.
    """

    def __init__(self, wfs, features, **kwargs):
        """
        Parameters
        ----------
        wfs: string or :class:`owslib.wfs.WebFeatureService` instance
            The WebFeatureService instance, or URL of a WFS service, from which
            to retrieve the geometries.
        features: string or list of strings
            The typename(s) of features available from the web service that
            will be retrieved. Somewhat analogous to layers in WMS/WMTS.

        Other Parameters
        ----------------
        **kwargs
            Keyword arguments to be used when drawing this feature.

        """
        try:
            from cartopy.io.ogc_clients import WFSGeometrySource
        except ImportError as e:
            raise ImportError('WFSFeature requires additional dependencies. If installed via pip, try `pip install cartopy[ows]`.\n') from e
        self.source = WFSGeometrySource(wfs, features)
        crs = self.source.default_projection()
        super().__init__(crs, **kwargs)
        self._kwargs.setdefault('edgecolor', 'black')
        self._kwargs.setdefault('facecolor', 'none')

    def geometries(self):
        min_x, min_y, max_x, max_y = self.crs.boundary.bounds
        geoms = self.source.fetch_geometries(self.crs, extent=(min_x, max_x, min_y, max_y))
        return iter(geoms)

    def intersecting_geometries(self, extent):
        geoms = self.source.fetch_geometries(self.crs, extent)
        return iter(geoms)