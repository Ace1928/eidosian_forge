import ssl
import socket
from os_ken import cfg
from os_ken.base import app_manager
from os_ken.lib import hub
from os_ken.lib import ip
from os_ken.services.protocols.ovsdb import client
from os_ken.services.protocols.ovsdb import event
from os_ken.controller import handler
def _start_remote(self, sock, client_address):
    schema_tables = cfg.CONF.ovsdb.schema_tables
    schema_ex_col = {}
    if cfg.CONF.ovsdb.schema_exclude_columns:
        for c in cfg.CONF.ovsdb.schema_exclude_columns:
            tbl, col = c.split('.')
            if tbl in schema_ex_col:
                schema_ex_col[tbl].append(col)
            else:
                schema_ex_col[tbl] = [col]
    app = client.RemoteOvsdb.factory(sock, client_address, probe_interval=self._probe_interval, min_backoff=self._min_backoff, max_backoff=self._max_backoff, schema_tables=schema_tables, schema_exclude_columns=schema_ex_col)
    if app:
        self._clients[app.name] = app
        app.start()
        ev = event.EventNewOVSDBConnection(app)
        self.send_event_to_observers(ev)
    else:
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except:
            pass
        sock.close()