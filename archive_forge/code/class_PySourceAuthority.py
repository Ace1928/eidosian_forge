import os
import time
from twisted.internet import defer
from twisted.names import common, dns, error
from twisted.python import failure
from twisted.python.compat import execfile, nativeString
from twisted.python.filepath import FilePath
class PySourceAuthority(FileAuthority):
    """
    A FileAuthority that is built up from Python source code.
    """

    def loadFile(self, filename):
        g, l = (self.setupConfigNamespace(), {})
        execfile(filename, g, l)
        if 'zone' not in l:
            raise ValueError('No zone defined in ' + filename)
        self.records = {}
        for rr in l['zone']:
            if isinstance(rr[1], dns.Record_SOA):
                self.soa = rr
            self.records.setdefault(rr[0].lower(), []).append(rr[1])

    def wrapRecord(self, type):

        def wrapRecordFunc(name, *arg, **kw):
            return (dns.domainString(name), type(*arg, **kw))
        return wrapRecordFunc

    def setupConfigNamespace(self):
        r = {}
        items = dns.__dict__.keys()
        for record in [x for x in items if x.startswith('Record_')]:
            type = getattr(dns, record)
            f = self.wrapRecord(type)
            r[record[len('Record_'):]] = f
        return r