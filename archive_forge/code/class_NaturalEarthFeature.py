from abc import ABCMeta, abstractmethod
import numpy as np
import shapely.geometry as sgeom
import cartopy.crs
import cartopy.io.shapereader as shapereader
class NaturalEarthFeature(Feature):
    """
    A simple interface to Natural Earth shapefiles.

    See https://www.naturalearthdata.com/

    """

    def __init__(self, category, name, scale, **kwargs):
        """
        Parameters
        ----------
        category
            The category of the dataset, i.e. either 'cultural' or 'physical'.
        name
            The name of the dataset, e.g. 'admin_0_boundary_lines_land'.
        scale
            The dataset scale, i.e. one of '10m', '50m', or '110m',
            or Scaler object. Dataset scales correspond to 1:10,000,000,
            1:50,000,000, and 1:110,000,000 respectively.

        Other Parameters
        ----------------
        **kwargs
            Keyword arguments to be used when drawing this feature.

        """
        super().__init__(cartopy.crs.PlateCarree(), **kwargs)
        self.category = category
        self.name = name
        if isinstance(scale, str):
            scale = Scaler(scale)
        self.scaler = scale
        self._validate_scale()

    @property
    def scale(self):
        return self.scaler.scale

    def _validate_scale(self):
        if self.scale not in ('110m', '50m', '10m'):
            raise ValueError(f'{self.scale!r} is not a valid Natural Earth scale. Valid scales are "110m", "50m", and "10m".')

    def geometries(self):
        """
        Returns an iterator of (shapely) geometries for this feature.

        """
        key = (self.name, self.category, self.scale)
        if key not in _NATURAL_EARTH_GEOM_CACHE:
            path = shapereader.natural_earth(resolution=self.scale, category=self.category, name=self.name)
            geometries = tuple(shapereader.Reader(path).geometries())
            _NATURAL_EARTH_GEOM_CACHE[key] = geometries
        else:
            geometries = _NATURAL_EARTH_GEOM_CACHE[key]
        return iter(geometries)

    def intersecting_geometries(self, extent):
        """
        Returns an iterator of shapely geometries that intersect with
        the given extent.
        The extent is assumed to be in the CRS of the feature.
        If extent is None, the method returns all geometries for this dataset.
        """
        self.scaler.scale_from_extent(extent)
        return super().intersecting_geometries(extent)

    def with_scale(self, new_scale):
        """
        Return a copy of the feature with a new scale.

        Parameters
        ----------
        new_scale
            The new dataset scale, i.e. one of '10m', '50m', or '110m'.
            Corresponding to 1:10,000,000, 1:50,000,000, and 1:110,000,000
            respectively.

        """
        return NaturalEarthFeature(self.category, self.name, new_scale, **self.kwargs)