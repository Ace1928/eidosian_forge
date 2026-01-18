import sys
from collections import OrderedDict
import numpy as np
from holoviews.core.data import Interface, DictInterface, MultiInterface
from holoviews.core.data.interface import DataError
from holoviews.core.data.spatialpandas import to_geom_dict
from holoviews.core.dimension import dimension_name
from holoviews.core.util import isscalar
from ..util import asarray, geom_types, geom_to_array, geom_length
class GeomDictInterface(DictInterface):
    datatype = 'geom_dictionary'

    @classmethod
    def applies(cls, obj):
        if 'shapely' not in sys.modules:
            return False
        return isinstance(obj, cls.types) and 'geometry' in obj and isinstance(obj['geometry'], geom_types) or isinstance(obj, geom_types)

    @classmethod
    def init(cls, eltype, data, kdims, vdims):
        odict_types = (OrderedDict,)
        if kdims is None:
            kdims = eltype.kdims
        if vdims is None:
            vdims = eltype.vdims
        dimensions = [dimension_name(d) for d in kdims + vdims]
        if isinstance(data, geom_types):
            data = {'geometry': data}
        elif not isinstance(data, dict) or 'geometry' not in data:
            xdim, ydim = kdims[:2]
            from shapely.geometry import Point, LineString, Polygon, MultiPoint, MultiPolygon, MultiLineString, LinearRing
            data = to_geom_dict(eltype, data, kdims, vdims, GeomDictInterface)
            geom = data.get('geom_type') or MultiInterface.geom_type(eltype)
            poly = 'holes' in data or geom == 'Polygon'
            if poly:
                single_type, multi_type = (Polygon, MultiPolygon)
            elif geom == 'Line':
                single_type, multi_type = (LineString, MultiLineString)
            elif geom == 'Ring':
                single_type, multi_type = (LinearRing, MultiPolygon)
            else:
                single_type, multi_type = (Point, MultiPoint)
            data['geometry'] = geom_from_dict(data, xdim.name, ydim.name, single_type, multi_type)
        if not cls.applies(data):
            raise ValueError("GeomDictInterface only handles dictionary types containing a 'geometry' key and shapely geometry value.")
        unpacked = []
        for d, vals in data.items():
            if isinstance(d, tuple):
                vals = asarray(vals)
                if vals.shape == (0,):
                    for sd in d:
                        unpacked.append((sd, np.array([], dtype=vals.dtype)))
                elif not vals.ndim == 2 and vals.shape[1] == len(d):
                    raise ValueError('Values for %s dimensions did not have the expected shape.')
                else:
                    for i, sd in enumerate(d):
                        unpacked.append((sd, vals[:, i]))
            elif d not in dimensions:
                unpacked.append((d, vals))
            else:
                if not isscalar(vals):
                    vals = asarray(vals)
                    if not vals.ndim == 1 and d in dimensions:
                        raise ValueError('DictInterface expects data for each column to be flat.')
                unpacked.append((d, vals))
        if not cls.expanded([vs for d, vs in unpacked if d in dimensions and (not isscalar(vs))]):
            raise ValueError('DictInterface expects data to be of uniform shape.')
        if isinstance(data, odict_types):
            data.update(unpacked)
        else:
            data = OrderedDict(unpacked)
        return (data, {'kdims': kdims, 'vdims': vdims}, {})

    @classmethod
    def validate(cls, dataset, validate_vdims):
        from shapely.geometry.base import BaseGeometry
        geom_dims = cls.geom_dims(dataset)
        if len(geom_dims) != 2:
            raise DataError('Expected %s instance to declare two key dimensions corresponding to the geometry coordinates but %d dimensions were found which did not refer to any columns.' % (type(dataset).__name__, len(geom_dims)), cls)
        elif 'geometry' not in dataset.data:
            raise DataError("Could not find a 'geometry' column in the data.")
        elif not isinstance(dataset.data['geometry'], BaseGeometry):
            raise DataError("The 'geometry' column should be a shapelygeometry type, found %s type instead." % type(dataset.data['geometry']).__name__)

    @classmethod
    def shape(cls, dataset):
        return (cls.length(dataset), len(dataset.dimensions()))

    @classmethod
    def geom_type(cls, dataset):
        from shapely.geometry import Polygon, MultiPolygon, LineString, MultiLineString, LinearRing
        geom = dataset.data['geometry']
        if isinstance(geom, (Polygon, MultiPolygon)):
            return 'Polygon'
        elif isinstance(geom, LinearRing):
            return 'Ring'
        elif isinstance(geom, (LineString, MultiLineString)):
            return 'Line'
        else:
            return 'Point'

    @classmethod
    def geo_column(cls, dataset):
        return 'geometry'

    @classmethod
    def has_holes(cls, dataset):
        from shapely.geometry import Polygon, MultiPolygon
        geom = dataset.data['geometry']
        if isinstance(geom, Polygon) and geom.interiors:
            return True
        elif isinstance(geom, MultiPolygon):
            for g in geom.geoms:
                if isinstance(g, Polygon) and g.interiors:
                    return True
        return False

    @classmethod
    def holes(cls, dataset):
        from shapely.geometry import Polygon, MultiPolygon
        geom = dataset.data['geometry']
        if isinstance(geom, Polygon):
            return [[[geom_to_array(h) for h in geom.interiors]]]
        elif isinstance(geom, MultiPolygon):
            return [[[geom_to_array(h) for h in g.interiors] for g in geom.geoms]]
        return []

    @classmethod
    def dimension_type(cls, dataset, dim):
        name = dataset.get_dimension(dim, strict=True).name
        if name in cls.geom_dims(dataset):
            return float
        values = dataset.data[name]
        return type(values) if isscalar(values) else values.dtype.type

    @classmethod
    def dtype(cls, dataset, dimension):
        name = dataset.get_dimension(dimension, strict=True).name
        if name in cls.geom_dims(dataset):
            return np.dtype('float')
        return Interface.dtype(dataset, dimension)

    @classmethod
    def range(cls, dataset, dim):
        dim = dataset.get_dimension(dim)
        geom_dims = cls.geom_dims(dataset)
        if dim in geom_dims:
            bounds = dataset.data['geometry'].bounds
            if not bounds:
                return (np.nan, np.nan)
            elif geom_dims.index(dim) == 0:
                return (bounds[0], bounds[2])
            else:
                return (bounds[1], bounds[3])
        else:
            return DictInterface.range(dataset, dim)

    @classmethod
    def length(cls, dataset):
        return geom_length(dataset.data['geometry'])

    @classmethod
    def geom_dims(cls, dataset):
        return [d for d in dataset.kdims + dataset.vdims if d.name not in dataset.data]

    @classmethod
    def values(cls, dataset, dim, expanded=True, flat=True, compute=True, keep_index=False):
        d = dataset.get_dimension(dim)
        geom_dims = cls.geom_dims(dataset)
        if d in geom_dims:
            g = dataset.data['geometry']
            if not g:
                return np.array([])
            array = geom_to_array(g)
            idx = geom_dims.index(d)
            return array[:, idx]
        return DictInterface.values(dataset, dim, expanded, flat)

    @classmethod
    def select(cls, dataset, selection_mask=None, **selection):
        if cls.geom_dims(dataset):
            data = cls.shape_mask(dataset, selection)
        else:
            data = dataset.data
        if selection_mask is None:
            selection_mask = cls.select_mask(dataset, selection)
        empty = not selection_mask.sum()
        dimensions = dataset.dimensions()
        if empty:
            return {d.name: np.array([], dtype=cls.dtype(dataset, d)) for d in dimensions}
        indexed = cls.indexed(dataset, selection)
        new_data = {}
        for k, v in data.items():
            if k not in dimensions or isscalar(v):
                new_data[k] = v
            else:
                new_data[k] = v[selection_mask]
        if indexed and len(next(iter(new_data.values()))) == 1 and (len(dataset.vdims) == 1):
            value = new_data[dataset.vdims[0].name]
            return value if isscalar(value) else value[0]
        return new_data

    @classmethod
    def shape_mask(cls, dataset, selection):
        xdim, ydim = cls.geom_dims(dataset)
        xsel = selection.pop(xdim.name, None)
        ysel = selection.pop(ydim.name, None)
        if xsel is None and ysel is None:
            return dataset.data
        from shapely.geometry import box
        if xsel is None:
            x0, x1 = cls.range(dataset, xdim)
        elif isinstance(xsel, slice):
            x0, x1 = (xsel.start, xsel.stop)
        elif isinstance(xsel, tuple):
            x0, x1 = xsel
        else:
            raise ValueError(f'Only slicing is supported on geometries, {xdim} selection is of type {type(xsel).__name__}.')
        if ysel is None:
            y0, y1 = cls.range(dataset, ydim)
        elif isinstance(ysel, slice):
            y0, y1 = (ysel.start, ysel.stop)
        elif isinstance(ysel, tuple):
            y0, y1 = ysel
        else:
            raise ValueError(f'Only slicing is supported on geometries, {ydim} selection is of type {type(ysel).__name__}.')
        bounds = box(x0, y0, x1, y1)
        geom = dataset.data['geometry']
        geom = geom.intersection(bounds)
        new_data = dict(dataset.data, geometry=geom)
        return new_data

    @classmethod
    def iloc(cls, dataset, index):
        from shapely.geometry import MultiPoint
        rows, cols = index
        data = dict(dataset.data)
        geom = data['geometry']
        if isinstance(geom, MultiPoint):
            if isscalar(rows) or isinstance(rows, slice):
                geom = geom.geoms[rows]
            elif isinstance(rows, (set, list)):
                geom = MultiPoint([geom.geoms[r] for r in rows])
        data['geometry'] = geom
        return data

    @classmethod
    def sample(cls, dataset, samples=None):
        if samples is None:
            samples = []
        raise NotImplementedError('sampling operation not implemented for geometries.')

    @classmethod
    def aggregate(cls, dataset, kdims, function, **kwargs):
        raise NotImplementedError('aggregate operation not implemented for geometries.')

    @classmethod
    def concat(cls, datasets, dimensions, vdims):
        raise NotImplementedError('concat operation not implemented for geometries.')