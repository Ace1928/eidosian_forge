import warnings
from contextlib import contextmanager
import pandas as pd
import shapely
import shapely.wkb
from geopandas import GeoDataFrame
from geopandas import _compat as compat
def _psql_insert_copy(tbl, conn, keys, data_iter):
    import io
    import csv
    s_buf = io.StringIO()
    writer = csv.writer(s_buf)
    writer.writerows(data_iter)
    s_buf.seek(0)
    columns = ', '.join(('"{}"'.format(k) for k in keys))
    dbapi_conn = conn.connection
    with dbapi_conn.cursor() as cur:
        sql = 'COPY "{}"."{}" ({}) FROM STDIN WITH CSV'.format(tbl.table.schema, tbl.table.name, columns)
        cur.copy_expert(sql=sql, file=s_buf)