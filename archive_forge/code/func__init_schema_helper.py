import logging
import operator
import os
import re
import sys
import weakref
import ovs.db.data
import ovs.db.parser
import ovs.db.schema
import ovs.db.types
import ovs.poller
import ovs.json
from ovs import jsonrpc
from ovs import ovsuuid
from ovs import stream
from ovs.db import idl
from os_ken.lib import hub
from os_ken.lib import ip
from os_ken.lib.ovs import vswitch_idl
from os_ken.lib.stringify import StringifyMixin
def _init_schema_helper(self):
    if self.schema_json is None:
        self.schema_json = self._rpc_get_schema_json(vswitch_idl.OVSREC_DB_NAME)
        schema_helper = idl.SchemaHelper(None, self.schema_json)
        schema_helper.register_all()
        self.schema = schema_helper.get_idl_schema()
    self.schema_helper = idl.SchemaHelper(None, self.schema_json)