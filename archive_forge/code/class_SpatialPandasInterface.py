import sys
from collections import defaultdict
import numpy as np
import pandas as pd
from ..dimension import dimension_name
from ..util import isscalar, unique_array, unique_iterator
from .interface import DataError, Interface
from .multipath import MultiInterface, ensure_ring
from .pandas import PandasInterface
class SpatialPandasInterface(MultiInterface):
    base_interface = PandasInterface
    datatype = 'spatialpandas'
    multi = True
    types = ()

    @classmethod
    def loaded(cls):
        return 'spatialpandas' in sys.modules

    @classmethod
    def applies(cls, obj):
        if not cls.loaded():
            return False
        is_sdf = isinstance(obj, cls.data_types())
        if 'geopandas' in sys.modules and 'geoviews' not in sys.modules:
            import geopandas as gpd
            is_sdf |= isinstance(obj, (gpd.GeoDataFrame, gpd.GeoSeries))
        return is_sdf

    @classmethod
    def data_types(cls):
        from spatialpandas import GeoDataFrame, GeoSeries
        return (GeoDataFrame, GeoSeries, cls.array_type())

    @classmethod
    def array_type(cls):
        from spatialpandas.geometry import GeometryArray
        return GeometryArray

    @classmethod
    def series_type(cls):
        from spatialpandas import GeoSeries
        return GeoSeries

    @classmethod
    def frame_type(cls):
        from spatialpandas import GeoDataFrame
        return GeoDataFrame

    @classmethod
    def geo_column(cls, data):
        col = 'geometry'
        stypes = cls.series_type()
        if col in data and isinstance(data[col], stypes):
            return col
        cols = [c for c in data.columns if isinstance(data[c], stypes)]
        if not cols:
            raise ValueError('No geometry column found in spatialpandas.GeoDataFrame, use the PandasInterface instead.')
        return cols[0]

    @classmethod
    def init(cls, eltype, data, kdims, vdims):
        from spatialpandas import GeoDataFrame
        if kdims is None:
            kdims = eltype.kdims
        if vdims is None:
            vdims = eltype.vdims
        if isinstance(data, cls.series_type()):
            data = data.to_frame()
        if 'geopandas' in sys.modules:
            import geopandas as gpd
            if isinstance(data, gpd.GeoSeries):
                data = data.to_frame()
            if isinstance(data, gpd.GeoDataFrame):
                data = GeoDataFrame(data)
        if isinstance(data, list):
            if 'shapely' in sys.modules:
                data = from_shapely(data)
            if isinstance(data, list):
                data = from_multi(eltype, data, kdims, vdims)
        elif isinstance(data, cls.array_type()):
            data = GeoDataFrame({'geometry': data})
        elif not isinstance(data, cls.frame_type()):
            raise ValueError(f'{cls.__name__} only support spatialpandas DataFrames.')
        elif 'geometry' not in data:
            cls.geo_column(data)
        index_names = data.index.names if isinstance(data, pd.DataFrame) else [data.index.name]
        if index_names == [None]:
            index_names = ['index']
        for kd in kdims + vdims:
            kd = dimension_name(kd)
            if kd in data.columns:
                continue
            if any((kd == ('index' if name is None else name) for name in index_names)):
                data = data.reset_index()
                break
        return (data, {'kdims': kdims, 'vdims': vdims}, {})

    @classmethod
    def validate(cls, dataset, vdims=True):
        dim_types = 'key' if vdims else 'all'
        geom_dims = cls.geom_dims(dataset)
        if len(geom_dims) != 2:
            raise DataError('Expected %s instance to declare two key dimensions corresponding to the geometry coordinates but %d dimensions were found which did not refer to any columns.' % (type(dataset).__name__, len(geom_dims)), cls)
        not_found = [d.name for d in dataset.dimensions(dim_types) if d not in geom_dims and d.name not in dataset.data]
        if not_found:
            raise DataError('Supplied data does not contain specified dimensions, the following dimensions were not found: %s' % repr(not_found), cls)

    @classmethod
    def dtype(cls, dataset, dimension):
        dim = dataset.get_dimension(dimension, strict=True)
        if dim in cls.geom_dims(dataset):
            col = cls.geo_column(dataset.data)
            return dataset.data[col].dtype.subtype
        return dataset.data[dim.name].dtype

    @classmethod
    def has_holes(cls, dataset):
        from spatialpandas.geometry import MultiPolygon, MultiPolygonDtype, Polygon, PolygonDtype
        col = cls.geo_column(dataset.data)
        series = dataset.data[col]
        if not isinstance(series.dtype, (MultiPolygonDtype, PolygonDtype)):
            return False
        for geom in series:
            if isinstance(geom, Polygon) and len(geom.data) > 1:
                return True
            elif isinstance(geom, MultiPolygon):
                for p in geom.data:
                    if len(p) > 1:
                        return True
        return False

    @classmethod
    def holes(cls, dataset):
        holes = []
        if not len(dataset.data):
            return holes
        col = cls.geo_column(dataset.data)
        series = dataset.data[col]
        return [geom_to_holes(geom) for geom in series]

    @classmethod
    def select(cls, dataset, selection_mask=None, **selection):
        xdim, ydim = cls.geom_dims(dataset)
        selection.pop(xdim.name, None)
        selection.pop(ydim.name, None)
        df = dataset.data
        if not selection:
            return df
        elif selection_mask is None:
            selection_mask = cls.select_mask(dataset, selection)
        indexed = cls.indexed(dataset, selection)
        df = df[selection_mask]
        if indexed and len(df) == 1 and (len(dataset.vdims) == 1):
            return df[dataset.vdims[0].name].iloc[0]
        return df

    @classmethod
    def select_mask(cls, dataset, selection):
        return cls.base_interface.select_mask(dataset, selection)

    @classmethod
    def geom_dims(cls, dataset):
        return [d for d in dataset.kdims + dataset.vdims if d.name not in dataset.data]

    @classmethod
    def dimension_type(cls, dataset, dim):
        dim = dataset.get_dimension(dim)
        return cls.dtype(dataset, dim).type

    @classmethod
    def isscalar(cls, dataset, dim, per_geom=False):
        """
        Tests if dimension is scalar in each subpath.
        """
        dim = dataset.get_dimension(dim)
        if dim in cls.geom_dims(dataset):
            return False
        elif per_geom:
            return all((isscalar(v) or len(list(unique_array(v))) == 1 for v in dataset.data[dim.name]))
        dim = dataset.get_dimension(dim)
        return len(dataset.data[dim.name].unique()) == 1

    @classmethod
    def range(cls, dataset, dim):
        dim = dataset.get_dimension(dim)
        geom_dims = cls.geom_dims(dataset)
        if dim in geom_dims:
            col = cls.geo_column(dataset.data)
            idx = geom_dims.index(dim)
            bounds = dataset.data[col].total_bounds
            if idx == 0:
                return (bounds[0], bounds[2])
            else:
                return (bounds[1], bounds[3])
        else:
            return cls.base_interface.range(dataset, dim)

    @classmethod
    def groupby(cls, dataset, dimensions, container_type, group_type, **kwargs):
        geo_dims = cls.geom_dims(dataset)
        if any((d in geo_dims for d in dimensions)):
            raise DataError('SpatialPandasInterface does not allow grouping by geometry dimension.', cls)
        return cls.base_interface.groupby(dataset, dimensions, container_type, group_type, **kwargs)

    @classmethod
    def aggregate(cls, columns, dimensions, function, **kwargs):
        raise NotImplementedError

    @classmethod
    def sample(cls, columns, samples=None):
        if samples is None:
            samples = []
        raise NotImplementedError

    @classmethod
    def reindex(cls, dataset, kdims=None, vdims=None):
        return dataset.data

    @classmethod
    def shape(cls, dataset):
        return (cls.length(dataset), len(dataset.dimensions()))

    @classmethod
    def sort(cls, dataset, by=None, reverse=False):
        if by is None:
            by = []
        geo_dims = cls.geom_dims(dataset)
        if any((d in geo_dims for d in by)):
            raise DataError('SpatialPandasInterface does not allow sorting by geometry dimension.', cls)
        return cls.base_interface.sort(dataset, by, reverse)

    @classmethod
    def length(cls, dataset):
        from spatialpandas.geometry import MultiPointDtype, Point
        col_name = cls.geo_column(dataset.data)
        column = dataset.data[col_name]
        geom_type = cls.geom_type(dataset)
        if not isinstance(column.dtype, MultiPointDtype) and geom_type != 'Point':
            return cls.base_interface.length(dataset)
        length = 0
        for geom in column:
            if isinstance(geom, Point):
                length += 1
            else:
                length += len(geom.buffer_values) // 2
        return length

    @classmethod
    def nonzero(cls, dataset):
        return bool(len(dataset.data.head(1)))

    @classmethod
    def redim(cls, dataset, dimensions):
        return cls.base_interface.redim(dataset, dimensions)

    @classmethod
    def add_dimension(cls, dataset, dimension, dim_pos, values, vdim):
        data = dataset.data.copy()
        geom_col = cls.geo_column(dataset.data)
        if dim_pos >= list(data.columns).index(geom_col):
            dim_pos -= 1
        if dimension.name not in data:
            data.insert(dim_pos, dimension.name, values)
        return data

    @classmethod
    def iloc(cls, dataset, index):
        from spatialpandas import GeoSeries
        from spatialpandas.geometry import MultiPointDtype
        rows, cols = index
        geom_dims = cls.geom_dims(dataset)
        geom_col = cls.geo_column(dataset.data)
        scalar = False
        columns = list(dataset.data.columns)
        if isinstance(cols, slice):
            cols = [d.name for d in dataset.dimensions()][cols]
        elif np.isscalar(cols):
            scalar = np.isscalar(rows)
            cols = [dataset.get_dimension(cols).name]
        else:
            cols = [dataset.get_dimension(d).name for d in index[1]]
        if not all((d in cols for d in geom_dims)):
            raise DataError('Cannot index a dimension which is part of the geometry column of a spatialpandas DataFrame.', cls)
        cols = list(unique_iterator([columns.index(geom_col) if c in geom_dims else columns.index(c) for c in cols]))
        if not isinstance(dataset.data[geom_col].dtype, MultiPointDtype):
            if scalar:
                return dataset.data.iloc[rows[0], cols[0]]
            elif isscalar(rows):
                rows = [rows]
            return dataset.data.iloc[rows, cols]
        geoms = dataset.data[geom_col]
        count = 0
        new_geoms, indexes = ([], [])
        for i, geom in enumerate(geoms):
            length = int(len(geom.buffer_values) / 2)
            if np.isscalar(rows):
                if count <= rows < count + length:
                    idx = (rows - count) * 2
                    data = geom.buffer_values[idx:idx + 2]
                    new_geoms.append(type(geom)(data))
                    indexes.append(i)
                    break
            elif isinstance(rows, slice):
                if rows.start is not None and rows.start > count + length:
                    continue
                elif rows.stop is not None and rows.stop < count:
                    break
                start = None if rows.start is None else max(rows.start - count, 0) * 2
                stop = None if rows.stop is None else min(rows.stop - count, length) * 2
                if rows.step is not None:
                    dataset.param.warning('.iloc step slicing currently not supported forthe multi-tabular data format.')
                sliced = geom.buffer_values[start:stop]
                if len(sliced):
                    indexes.append(i)
                    new_geoms.append(type(geom)(sliced))
            else:
                sub_rows = [v for r in rows for v in ((r - count) * 2, (r - count) * 2 + 1) if count <= r < count + length]
                if sub_rows:
                    indexes.append(i)
                    idxs = np.array(sub_rows, dtype=int)
                    new_geoms.append(type(geom)(geom.buffer_values[idxs]))
            count += length
        new = dataset.data.iloc[indexes].copy()
        new[geom_col] = GeoSeries(new_geoms)
        return new

    @classmethod
    def values(cls, dataset, dimension, expanded=True, flat=True, compute=True, keep_index=False):
        dimension = dataset.get_dimension(dimension)
        geom_dims = dataset.interface.geom_dims(dataset)
        data = dataset.data
        isgeom = dimension in geom_dims
        geom_col = cls.geo_column(dataset.data)
        is_points = cls.geom_type(dataset) == 'Point'
        if isgeom and keep_index:
            return data[geom_col]
        elif not isgeom:
            if is_points:
                return data[dimension.name].values
            return get_value_array(data, dimension, expanded, keep_index, geom_col, is_points)
        elif not len(data):
            return np.array([])
        geom_type = cls.geom_type(dataset)
        index = geom_dims.index(dimension)
        geom_series = data[geom_col]
        if compute and hasattr(geom_series, 'compute'):
            geom_series = geom_series.compute()
        return geom_array_to_array(geom_series.values, index, expanded, geom_type)

    @classmethod
    def split(cls, dataset, start, end, datatype, **kwargs):
        from spatialpandas import GeoDataFrame, GeoSeries
        from ...element import Polygons
        objs = []
        if not len(dataset.data):
            return []
        xdim, ydim = cls.geom_dims(dataset)
        value_dims = [dim for dim in dataset.kdims + dataset.vdims if dim not in (xdim, ydim)]
        row = dataset.data.iloc[0]
        col = cls.geo_column(dataset.data)
        geom_type = cls.geom_type(dataset)
        if datatype is not None:
            arr = geom_to_array(row[col], geom_type=geom_type)
            d = {(xdim.name, ydim.name): arr}
            d.update({dim.name: row[dim.name] for dim in value_dims})
            ds = dataset.clone(d, datatype=['dictionary'])
        holes = cls.holes(dataset) if cls.has_holes(dataset) else None
        for i, row in dataset.data.iterrows():
            if datatype is None:
                gdf = GeoDataFrame({c: GeoSeries([row[c]]) if c == 'geometry' else [row[c]] for c in dataset.data.columns})
                objs.append(dataset.clone(gdf))
                continue
            geom = row[col]
            gt = geom_type or get_geom_type(dataset.data, col)
            arr = geom_to_array(geom, geom_type=gt)
            d = {xdim.name: arr[:, 0], ydim.name: arr[:, 1]}
            d.update({dim.name: row[dim.name] for dim in value_dims})
            if datatype in ('dictionary', 'columns'):
                if holes is not None:
                    d[Polygons._hole_key] = holes[i]
                d['geom_type'] = gt
                objs.append(d)
                continue
            ds.data = d
            if datatype == 'array':
                obj = ds.array(**kwargs)
            elif datatype == 'dataframe':
                obj = ds.dframe(**kwargs)
            else:
                raise ValueError(f'{datatype} datatype not support')
            objs.append(obj)
        return objs

    @classmethod
    def dframe(cls, dataset, dimensions):
        if dimensions:
            return dataset.data[dimensions]
        else:
            return dataset.data.copy()

    @classmethod
    def as_dframe(cls, dataset):
        return dataset.data