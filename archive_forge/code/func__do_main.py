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
def _do_main(self, commands):
    """
        :type commands: list of VSCtlCommand
        """
    self._reset()
    self._init_schema_helper()
    self._run_prerequisites(commands)
    idl_ = idl.Idl(self.remote, self.schema_helper)
    seqno = idl_.change_seqno
    while True:
        self._idl_wait(idl_, seqno)
        seqno = idl_.change_seqno
        if self._do_vsctl(idl_, commands):
            break
        if self.txn:
            self.txn.abort()
            self.txn = None
    idl_.close()