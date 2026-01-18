import uuid
from os_ken.lib import dpid as dpidlib
from os_ken.services.protocols.ovsdb import event as ovsdb_event
def _get_iface(tables, attr_val, attr_name='name'):
    return _get_table_row('Interface', attr_name, attr_val, tables=tables)