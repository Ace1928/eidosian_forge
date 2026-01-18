from eventlet.patcher import slurp_properties
from eventlet import tpool
def Connection(*args, **kw):
    conn = tpool.execute(__orig_connections.Connection, *args, **kw)
    return tpool.Proxy(conn, autowrap_names=('cursor',))