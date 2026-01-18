import collections
import enum
import functools
import uuid
import ovs.db.data as data
import ovs.db.parser
import ovs.db.schema
import ovs.jsonrpc
import ovs.ovsuuid
import ovs.poller
import ovs.vlog
from ovs.db import custom_index
from ovs.db import error
def __parse_update(self, update, version, tables=None):
    try:
        if not tables:
            self.__do_parse_update(update, version, self.tables)
        else:
            self.__do_parse_update(update, version, tables)
    except error.Error as e:
        vlog.err('%s: error parsing update: %s' % (self._session.get_name(), e))