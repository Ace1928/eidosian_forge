import codecs
import errno
import os
import random
import sys
import ovs.json
import ovs.poller
import ovs.reconnect
import ovs.stream
import ovs.timeval
import ovs.util
import ovs.vlog
def __log_msg(self, title, msg):
    if vlog.dbg_is_enabled():
        vlog.dbg('%s: %s %s' % (self.name, title, msg))