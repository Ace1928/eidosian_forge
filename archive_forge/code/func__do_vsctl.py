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
def _do_vsctl(self, idl_, commands):
    self.txn = idl.Transaction(idl_)
    if self.dry_run:
        self.txn.dry_run = True
    self.txn.add_comment('ovs-vsctl')
    ovs_rows = idl_.tables[vswitch_idl.OVSREC_TABLE_OPEN_VSWITCH].rows
    if ovs_rows:
        ovs_ = list(ovs_rows.values())[0]
    else:
        ovs_ = self.txn.insert(idl_.tables[vswitch_idl.OVSREC_TABLE_OPEN_VSWITCH])
    if self.wait_for_reload:
        ovs_.increment(vswitch_idl.OVSREC_OPEN_VSWITCH_COL_NEXT_CFG)
    ctx = VSCtlContext(idl_, self.txn, ovs_)
    for command in commands:
        if not command._run:
            continue
        command._run(ctx, command)
        if ctx.try_again:
            return False
    LOG.debug('result:\n%s', [command.result for command in commands])
    ctx.done()
    status = self.txn.commit_block()
    next_cfg = 0
    if self.wait_for_reload and status == idl.Transaction.SUCCESS:
        next_cfg = self.txn.get_increment_new_value()
    txn_ = self.txn
    self.txn = None
    if status in (idl.Transaction.UNCOMMITTED, idl.Transaction.INCOMPLETE):
        not_reached()
    elif status == idl.Transaction.ABORTED:
        vsctl_fatal('transaction aborted')
    elif status == idl.Transaction.UNCHANGED:
        LOG.debug('unchanged')
    elif status == idl.Transaction.SUCCESS:
        LOG.debug('success')
    elif status == idl.Transaction.TRY_AGAIN:
        return False
    elif status == idl.Transaction.ERROR:
        vsctl_fatal('transaction error: %s' % txn_.get_error())
    elif status == idl.Transaction.NOT_LOCKED:
        vsctl_fatal('database not locked')
    else:
        not_reached()
    if self.wait_for_reload and status != idl.Transaction.UNCHANGED:
        while True:
            idl_.run()
            if ovs_.cur_cfg >= next_cfg:
                break
            self._idl_block(idl_)
    return True