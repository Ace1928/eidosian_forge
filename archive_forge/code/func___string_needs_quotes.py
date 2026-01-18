import functools
import re
import uuid
import ovs.db.parser
import ovs.db.types
import ovs.json
import ovs.jsonrpc
import ovs.ovsuuid
import ovs.poller
import ovs.socket_util
from ovs.db import error
@staticmethod
def __string_needs_quotes(s):
    return Atom.__need_quotes_re.match(s)