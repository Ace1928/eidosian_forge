import sys
from twisted.internet import reactor
from twisted.internet.protocol import Factory
from twisted.protocols import basic
class POP3TestServer(basic.LineReceiver):

    def __init__(self, contextFactory=None):
        self.loggedIn = False
        self.caps = None
        self.tmpUser = None
        self.ctx = contextFactory

    def sendSTATResp(self, req):
        self.sendLine(STAT)

    def sendUIDLResp(self, req):
        self.sendLine(UIDL)

    def sendLISTResp(self, req):
        self.sendLine(LIST)

    def sendCapabilities(self):
        if self.caps is None:
            self.caps = [CAP_START]
        if UIDL_SUPPORT:
            self.caps.append(CAPABILITIES_UIDL)
        if SSL_SUPPORT:
            self.caps.append(CAPABILITIES_SSL)
        for cap in CAPABILITIES:
            self.caps.append(cap)
        resp = b'\r\n'.join(self.caps)
        resp += b'\r\n.'
        self.sendLine(resp)

    def connectionMade(self):
        if DENY_CONNECTION:
            self.disconnect()
            return
        if SLOW_GREETING:
            reactor.callLater(20, self.sendGreeting)
        else:
            self.sendGreeting()

    def sendGreeting(self):
        self.sendLine(CONNECTION_MADE)

    def lineReceived(self, line):
        """Error Conditions"""
        uline = line.upper()
        find = lambda s: uline.find(s) != -1
        if TIMEOUT_RESPONSE:
            return
        if DROP_CONNECTION:
            self.disconnect()
            return
        elif find(b'CAPA'):
            if INVALID_CAPABILITY_RESPONSE:
                self.sendLine(INVALID_RESPONSE)
            else:
                self.sendCapabilities()
        elif find(b'STLS') and SSL_SUPPORT:
            self.startTLS()
        elif find(b'USER'):
            if INVALID_LOGIN_RESPONSE:
                self.sendLine(INVALID_RESPONSE)
                return
            resp = None
            try:
                self.tmpUser = line.split(' ')[1]
                resp = VALID_RESPONSE
            except BaseException:
                resp = AUTH_DECLINED
            self.sendLine(resp)
        elif find(b'PASS'):
            resp = None
            try:
                pwd = line.split(' ')[1]
                if self.tmpUser is None or pwd is None:
                    resp = AUTH_DECLINED
                elif self.tmpUser == USER and pwd == PASS:
                    resp = AUTH_ACCEPTED
                    self.loggedIn = True
                else:
                    resp = AUTH_DECLINED
            except BaseException:
                resp = AUTH_DECLINED
            self.sendLine(resp)
        elif find(b'QUIT'):
            self.loggedIn = False
            self.sendLine(LOGOUT_COMPLETE)
            self.disconnect()
        elif INVALID_SERVER_RESPONSE:
            self.sendLine(INVALID_RESPONSE)
        elif not self.loggedIn:
            self.sendLine(NOT_LOGGED_IN)
        elif find(b'NOOP'):
            self.sendLine(VALID_RESPONSE)
        elif find(b'STAT'):
            if TIMEOUT_DEFERRED:
                return
            self.sendLine(STAT)
        elif find(b'LIST'):
            if TIMEOUT_DEFERRED:
                return
            self.sendLine(LIST)
        elif find(b'UIDL'):
            if TIMEOUT_DEFERRED:
                return
            elif not UIDL_SUPPORT:
                self.sendLine(INVALID_RESPONSE)
                return
            self.sendLine(UIDL)

    def startTLS(self):
        if SSL_SUPPORT and self.ctx is not None:
            self.sendLine(b'+OK Begin TLS negotiation now')
            self.transport.startTLS(self.ctx)
        else:
            self.sendLine(b'-ERR TLS not available')

    def disconnect(self):
        self.transport.loseConnection()