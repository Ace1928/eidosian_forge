from twisted.application import service
from twisted.internet import defer, task
from twisted.names import client, common, dns, resolve
from twisted.names.authority import FileAuthority
from twisted.python import failure, log
from twisted.python.compat import nativeString
def _cbZone(self, zone):
    ans, _, _ = zone
    self.records = r = {}
    for rec in ans:
        if not self.soa and rec.type == dns.SOA:
            self.soa = (rec.name.name.lower(), rec.payload)
        else:
            r.setdefault(rec.name.name.lower(), []).append(rec.payload)