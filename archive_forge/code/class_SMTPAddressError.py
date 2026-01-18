from typing import Optional
class SMTPAddressError(SMTPServerError):

    def __init__(self, addr, code, resp):
        from twisted.mail.smtp import Address
        SMTPServerError.__init__(self, code, resp)
        self.addr = Address(addr)

    def __str__(self) -> str:
        return '%.3d <%s>... %s' % (self.code, self.addr, self.resp)