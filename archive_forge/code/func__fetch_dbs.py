import logging
import os
from ovs import jsonrpc
from ovs import stream
from ovs import util as ovs_util
from ovs.db import schema
def _fetch_dbs(self, rpc):
    request = jsonrpc.Message.create_request('list_dbs', [])
    error, reply = rpc.transact_block(request)
    self._check_txn(error, reply)
    dbs = set()
    for name in reply.result:
        dbs.add(name)
    return dbs