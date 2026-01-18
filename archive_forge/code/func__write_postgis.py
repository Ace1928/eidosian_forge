import warnings
from contextlib import contextmanager
import pandas as pd
import shapely
import shapely.wkb
from geopandas import GeoDataFrame
from geopandas import _compat as compat
def _write_postgis(gdf, name, con, schema=None, if_exists='fail', index=False, index_label=None, chunksize=None, dtype=None):
    """
    Upload GeoDataFrame into PostGIS database.

    This method requires SQLAlchemy and GeoAlchemy2, and a PostgreSQL
    Python driver (e.g. psycopg2) to be installed.

    Parameters
    ----------
    name : str
        Name of the target table.
    con : sqlalchemy.engine.Connection or sqlalchemy.engine.Engine
        Active connection to the PostGIS database.
    if_exists : {'fail', 'replace', 'append'}, default 'fail'
        How to behave if the table already exists:

        - fail: Raise a ValueError.
        - replace: Drop the table before inserting new values.
        - append: Insert new values to the existing table.
    schema : string, optional
        Specify the schema. If None, use default schema: 'public'.
    index : bool, default True
        Write DataFrame index as a column.
        Uses *index_label* as the column name in the table.
    index_label : string or sequence, default None
        Column label for index column(s).
        If None is given (default) and index is True,
        then the index names are used.
    chunksize : int, optional
        Rows will be written in batches of this size at a time.
        By default, all rows will be written at once.
    dtype : dict of column name to SQL type, default None
        Specifying the datatype for columns.
        The keys should be the column names and the values
        should be the SQLAlchemy types.

    Examples
    --------

    >>> from sqlalchemy import create_engine  # doctest: +SKIP
    >>> engine = create_engine("postgresql://myusername:mypassword@myhost:5432/mydatabase";)  # doctest: +SKIP
    >>> gdf.to_postgis("my_table", engine)  # doctest: +SKIP
    """
    try:
        from geoalchemy2 import Geometry
        from sqlalchemy import text
    except ImportError:
        raise ImportError("'to_postgis()' requires geoalchemy2 package.")
    gdf = gdf.copy()
    geom_name = gdf.geometry.name
    srid = _get_srid_from_crs(gdf)
    geometry_type, has_curve = _get_geometry_type(gdf)
    if dtype is not None:
        dtype[geom_name] = Geometry(geometry_type=geometry_type, srid=srid)
    else:
        dtype = {geom_name: Geometry(geometry_type=geometry_type, srid=srid)}
    if has_curve:
        gdf = _convert_linearring_to_linestring(gdf, geom_name)
    gdf = _convert_to_ewkb(gdf, geom_name, srid)
    if schema is not None:
        schema_name = schema
    else:
        schema_name = 'public'
    if if_exists == 'append':
        with _get_conn(con) as connection:
            if connection.dialect.has_table(connection, name, schema):
                target_srid = connection.execute(text("SELECT Find_SRID('{schema}', '{table}', '{geom_col}');".format(schema=schema_name, table=name, geom_col=geom_name))).fetchone()[0]
                if target_srid != srid:
                    msg = 'The CRS of the target table (EPSG:{epsg_t}) differs from the CRS of current GeoDataFrame (EPSG:{epsg_src}).'.format(epsg_t=target_srid, epsg_src=srid)
                    raise ValueError(msg)
    with _get_conn(con) as connection:
        gdf.to_sql(name, connection, schema=schema_name, if_exists=if_exists, index=index, index_label=index_label, chunksize=chunksize, dtype=dtype, method=_psql_insert_copy)