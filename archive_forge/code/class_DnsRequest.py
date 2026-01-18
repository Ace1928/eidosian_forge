import socket, string, types, time, select
import errno
from . import Type,Class,Opcode
from . import Lib
class DnsRequest:
    """ high level Request object """

    def __init__(self, *name, **args):
        self.donefunc = None
        self.defaults = {}
        self.argparse(name, args)
        self.defaults = self.args
        self.tid = 0
        self.resulttype = ''
        if len(self.defaults['server']) == 0:
            raise DNSError('No working name servers discovered')

    def argparse(self, name, args):
        if not name and 'name' in self.defaults:
            args['name'] = self.defaults['name']
        if type(name) is bytes or type(name) is str:
            args['name'] = name
        elif len(name) == 1:
            if name[0]:
                args['name'] = name[0]
        if defaults['server_rotate'] and type(defaults['server']) == types.ListType:
            defaults['server'] = defaults['server'][1:] + defaults['server'][:1]
        for i in list(defaults.keys()):
            if i not in args:
                if i in self.defaults:
                    args[i] = self.defaults[i]
                else:
                    args[i] = defaults[i]
        if type(args['server']) == bytes or type(args['server']) == str:
            args['server'] = [args['server']]
        self.args = args

    def socketInit(self, a, b):
        self.s = socket.socket(a, b)

    def processUDPReply(self):
        if self.timeout > 0:
            r, w, e = select.select([self.s], [], [], self.timeout)
            if not len(r):
                raise TimeoutError('Timeout')
        self.reply, self.from_address = self.s.recvfrom(65535)
        self.time_finish = time.time()
        self.args['server'] = self.ns
        return self.processReply()

    def _readall(self, f, count):
        res = f.read(count)
        while len(res) < count:
            if self.timeout > 0:
                rem = self.time_start + self.timeout - time.time()
                if rem <= 0:
                    raise DNSError('Timeout')
                self.s.settimeout(rem)
            buf = f.read(count - len(res))
            if not buf:
                raise DNSError('incomplete reply - %d of %d read' % (len(res), count))
            res += buf
        return res

    def processTCPReply(self):
        if self.timeout > 0:
            self.s.settimeout(self.timeout)
        else:
            self.s.settimeout(None)
        f = self.s.makefile('rb')
        try:
            header = self._readall(f, 2)
            count = Lib.unpack16bit(header)
            self.reply = self._readall(f, count)
        finally:
            f.close()
        self.time_finish = time.time()
        self.args['server'] = self.ns
        return self.processReply()

    def processReply(self):
        self.args['elapsed'] = (self.time_finish - self.time_start) * 1000
        if not self.resulttype:
            u = Lib.Munpacker(self.reply)
        elif self.resulttype == 'default':
            u = Lib.MunpackerDefault(self.reply)
        elif self.resulttype == 'binary':
            u = Lib.MunpackerBinary(self.reply)
        elif self.resulttype == 'text':
            u = Lib.MunpackerText(self.reply)
        elif self.resulttype == 'integer':
            u = Lib.MunpackerInteger(self.reply)
        else:
            raise SyntaxError('Unknown resulttype: ' + self.resulttype)
        r = Lib.DnsResult(u, self.args)
        r.args = self.args
        return r

    def getSource(self):
        """Pick random source port to avoid DNS cache poisoning attack."""
        while True:
            try:
                source_port = random.randint(1024, 65535)
                self.s.bind(('', source_port))
                break
            except socket.error as msg:
                if msg.errno != errno.EADDRINUSE:
                    raise

    def conn(self):
        self.getSource()
        self.s.connect((self.ns, self.port))

    def qry(self, *name, **args):
        """
        Request function for the DnsRequest class.  In addition to standard
        DNS args, the special pydns arg 'resulttype' can optionally be passed.
        Valid resulttypes are 'default', 'text', 'decimal', and 'binary'.

        Defaults are configured to be compatible with pydns:
        AAAA: decimal
        Others: text
        """
        ' needs a refactoring '
        self.argparse(name, args)
        protocol = self.args['protocol']
        self.port = self.args['port']
        self.tid = random.randint(0, 65535)
        self.timeout = self.args['timeout']
        opcode = self.args['opcode']
        rd = self.args['rd']
        server = self.args['server']
        if 'resulttype' in self.args:
            self.resulttype = self.args['resulttype']
        else:
            self.resulttype = 'default'
        if type(self.args['qtype']) == bytes or type(self.args['qtype']) == str:
            try:
                qtype = getattr(Type, str(self.args['qtype'].upper()))
            except AttributeError:
                raise ArgumentError('unknown query type')
        else:
            qtype = self.args['qtype']
        if 'name' not in self.args:
            print(self.args)
            raise ArgumentError('nothing to lookup')
        qname = self.args['name']
        if qtype == Type.AXFR and protocol != 'tcp':
            print('Query type AXFR, protocol forced to TCP')
            protocol = 'tcp'
        m = Lib.Mpacker()
        m.addHeader(self.tid, 0, opcode, 0, 0, rd, 0, 0, 0, 1, 0, 0, 0)
        m.addQuestion(qname, qtype, Class.IN)
        self.request = m.getbuf()
        try:
            if protocol == 'udp':
                self.sendUDPRequest(server)
            else:
                self.sendTCPRequest(server)
        except socket.error as reason:
            raise SocketError(reason)
        return self.response

    def req(self, *name, **args):
        """ needs a refactoring """
        self.argparse(name, args)
        try:
            if self.args['resulttype']:
                raise ArgumentError('Restulttype {0} set with DNS.req, use DNS.qry to specify result type.'.format(self.args['resulttype']))
        except:
            pass
        protocol = self.args['protocol']
        self.port = self.args['port']
        self.tid = random.randint(0, 65535)
        self.timeout = self.args['timeout']
        opcode = self.args['opcode']
        rd = self.args['rd']
        server = self.args['server']
        if type(self.args['qtype']) == bytes or type(self.args['qtype']) == str:
            try:
                qtype = getattr(Type, str(self.args['qtype'].upper()))
            except AttributeError:
                raise ArgumentError('unknown query type')
        else:
            qtype = self.args['qtype']
        if 'name' not in self.args:
            print(self.args)
            raise ArgumentError('nothing to lookup')
        qname = self.args['name']
        if qtype == Type.AXFR and protocol != 'tcp':
            print('Query type AXFR, protocol forced to TCP')
            protocol = 'tcp'
        m = Lib.Mpacker()
        m.addHeader(self.tid, 0, opcode, 0, 0, rd, 0, 0, 0, 1, 0, 0, 0)
        m.addQuestion(qname, qtype, Class.IN)
        self.request = m.getbuf()
        try:
            if protocol == 'udp':
                self.sendUDPRequest(server)
            else:
                self.sendTCPRequest(server)
        except socket.error as reason:
            raise SocketError(reason)
        return self.response

    def sendUDPRequest(self, server):
        """refactor me"""
        first_socket_error = None
        self.response = None
        for self.ns in server:
            try:
                if self.ns.count(':'):
                    if hasattr(socket, 'has_ipv6') and socket.has_ipv6:
                        self.socketInit(socket.AF_INET6, socket.SOCK_DGRAM)
                    else:
                        continue
                else:
                    self.socketInit(socket.AF_INET, socket.SOCK_DGRAM)
                try:
                    self.time_start = time.time()
                    self.conn()
                    self.s.send(self.request)
                    r = self.processUDPReply()
                    while r.header['id'] != self.tid or self.from_address[1] != self.port:
                        r = self.processUDPReply()
                    self.response = r
                finally:
                    self.s.close()
            except socket.error as e:
                first_socket_error = first_socket_error or e
                continue
            except TimeoutError as t:
                first_socket_error = first_socket_error or t
                continue
            if self.response:
                break
        if not self.response and first_socket_error:
            raise first_socket_error

    def sendTCPRequest(self, server):
        """ do the work of sending a TCP request """
        first_socket_error = None
        self.response = None
        for self.ns in server:
            try:
                if self.ns.count(':'):
                    if hasattr(socket, 'has_ipv6') and socket.has_ipv6:
                        self.socketInit(socket.AF_INET6, socket.SOCK_STREAM)
                    else:
                        continue
                else:
                    self.socketInit(socket.AF_INET, socket.SOCK_STREAM)
                try:
                    self.time_start = time.time()
                    self.conn()
                    buf = Lib.pack16bit(len(self.request)) + self.request
                    self.s.setblocking(0)
                    self.s.sendall(buf)
                    r = self.processTCPReply()
                    if r.header['id'] == self.tid:
                        self.response = r
                        break
                finally:
                    self.s.close()
            except socket.error as e:
                first_socket_error = first_socket_error or e
                continue
            except TimeoutError as t:
                first_socket_error = first_socket_error or t
                continue
            if self.response:
                break
        if not self.response and first_socket_error:
            raise first_socket_error