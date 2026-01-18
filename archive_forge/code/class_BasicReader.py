import io
import itertools
from pathlib import Path
from urllib.error import HTTPError
import shapefile
import shapely.geometry as sgeom
from cartopy import config
from cartopy.io import Downloader
class BasicReader:
    """
    Provide an interface for accessing the contents of a shapefile with the
    Python Shapefile Library (PyShp). See the PyShp
    `Readme <https://pypi.org/project/pyshp/>`_ for more information.

    The primary methods used on a BasicReader instance are
    :meth:`~cartopy.io.shapereader.BasicReader.records` and
    :meth:`~cartopy.io.shapereader.BasicReader.geometries`.

    """

    def __init__(self, filename, bbox=None, **kwargs):
        self._reader = reader = shapefile.Reader(filename, **kwargs)
        self._bbox = bbox
        if reader.shp is None or reader.shx is None or reader.dbf is None:
            raise ValueError("Incomplete shapefile definition in '%s'." % filename)
        self._fields = self._reader.fields

    def close(self):
        return self._reader.close()

    def __len__(self):
        return len(self._reader)

    def geometries(self):
        """
        Return an iterator of shapely geometries from the shapefile.

        This interface is useful for accessing the geometries of the
        shapefile where knowledge of the associated metadata is not necessary.
        In the case where further metadata is needed use the
        :meth:`~cartopy.io.shapereader.BasicReader.records`
        interface instead, extracting the geometry from the record with the
        :meth:`~Record.geometry` method.

        """
        for shape in self._reader.iterShapes(bbox=self._bbox):
            if shape.shapeType != shapefile.NULL:
                yield sgeom.shape(shape)

    def records(self):
        """
        Return an iterator of :class:`~Record` instances.

        """
        fields = self._reader.fields[1:]
        for shape_record in self._reader.iterShapeRecords(bbox=self._bbox):
            attributes = shape_record.record.as_dict()
            yield Record(shape_record.shape, attributes, fields)