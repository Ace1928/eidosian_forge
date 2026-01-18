import types
import socket
from . import Type
from . import Class
from . import Opcode
from . import Status
import DNS
from .Base import DNSError
from struct import pack as struct_pack
from struct import unpack as struct_unpack
from socket import inet_ntoa, inet_aton, inet_ntop, AF_INET6
class DnsResult:

    def __init__(self, u, args):
        self.header = {}
        self.questions = []
        self.answers = []
        self.authority = []
        self.additional = []
        self.args = args
        self.storeM(u)

    def show(self):
        import time
        print('; <<>> PDG.py 1.0 <<>> %s %s' % (self.args['name'], self.args['qtype']))
        opt = ''
        if self.args['rd']:
            opt = opt + 'recurs '
        h = self.header
        print(';; options: ' + opt)
        print(';; got answer:')
        print(';; ->>HEADER<<- opcode %s, status %s, id %d' % (h['opcode'], h['status'], h['id']))
        flags = list(filter(lambda x, h=h: h[x], ('qr', 'aa', 'rd', 'ra', 'tc')))
        print(';; flags: %s; Ques: %d, Ans: %d, Auth: %d, Addit: %d' % (' '.join(flags), h['qdcount'], h['ancount'], h['nscount'], h['arcount']))
        print(';; QUESTIONS:')
        for q in self.questions:
            print(';;      %s, type = %s, class = %s' % (q['qname'], q['qtypestr'], q['qclassstr']))
        print()
        print(';; ANSWERS:')
        for a in self.answers:
            print('%-20s    %-6s  %-6s  %s' % (a['name'], repr(a['ttl']), a['typename'], a['data']))
        print()
        print(';; AUTHORITY RECORDS:')
        for a in self.authority:
            print('%-20s    %-6s  %-6s  %s' % (a['name'], repr(a['ttl']), a['typename'], a['data']))
        print()
        print(';; ADDITIONAL RECORDS:')
        for a in self.additional:
            print('%-20s    %-6s  %-6s  %s' % (a['name'], repr(a['ttl']), a['typename'], a['data']))
        print()
        if 'elapsed' in self.args:
            print(';; Total query time: %d msec' % self.args['elapsed'])
        print(';; To SERVER: %s' % self.args['server'])
        print(';; WHEN: %s' % time.ctime(time.time()))

    def storeM(self, u):
        self.header['id'], self.header['qr'], self.header['opcode'], self.header['aa'], self.header['tc'], self.header['rd'], self.header['ra'], self.header['z'], self.header['rcode'], self.header['qdcount'], self.header['ancount'], self.header['nscount'], self.header['arcount'] = u.getHeader()
        self.header['opcodestr'] = Opcode.opcodestr(self.header['opcode'])
        self.header['status'] = Status.statusstr(self.header['rcode'])
        for i in range(self.header['qdcount']):
            self.questions.append(self.storeQ(u))
        for i in range(self.header['ancount']):
            self.answers.append(self.storeRR(u))
        for i in range(self.header['nscount']):
            self.authority.append(self.storeRR(u))
        for i in range(self.header['arcount']):
            self.additional.append(self.storeRR(u))

    def storeQ(self, u):
        q = {}
        q['qname'], q['qtype'], q['qclass'] = u.getQuestion()
        q['qtypestr'] = Type.typestr(q['qtype'])
        q['qclassstr'] = Class.classstr(q['qclass'])
        return q

    def storeRR(self, u):
        r = {}
        r['name'], r['type'], r['class'], r['ttl'], r['rdlength'] = u.getRRheader()
        r['typename'] = Type.typestr(r['type'])
        r['classstr'] = Class.classstr(r['class'])
        mname = 'get%sdata' % r['typename']
        if hasattr(u, mname):
            r['data'] = getattr(u, mname)()
        else:
            r['data'] = u.getbytes(r['rdlength'])
        return r