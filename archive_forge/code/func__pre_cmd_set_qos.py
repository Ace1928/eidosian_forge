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
def _pre_cmd_set_qos(self, ctx, command):
    self._pre_get_info(ctx, command)
    schema_helper = self.schema_helper
    schema_helper.register_columns(vswitch_idl.OVSREC_TABLE_QOS, [vswitch_idl.OVSREC_QOS_COL_EXTERNAL_IDS, vswitch_idl.OVSREC_QOS_COL_OTHER_CONFIG, vswitch_idl.OVSREC_QOS_COL_QUEUES, vswitch_idl.OVSREC_QOS_COL_TYPE])