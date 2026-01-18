from ncclient import NCClientError
class SessionCloseError(TransportError):

    def __init__(self, in_buf, out_buf=None):
        msg = 'Unexpected session close'
        if in_buf:
            msg += '\nIN_BUFFER: `%s`' % in_buf
        if out_buf:
            msg += ' OUT_BUFFER: `%s`' % out_buf
        SSHError.__init__(self, msg)