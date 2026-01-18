import uuid
from os_ken.lib import dpid as dpidlib
from os_ken.services.protocols.ovsdb import event as ovsdb_event
def _get_table_row(table, attr_name, attr_value, tables):
    sentinel = object()
    for row in tables[table].rows.values():
        if getattr(row, attr_name, sentinel) == attr_value:
            return row