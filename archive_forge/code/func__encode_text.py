import binascii
import email.charset
import email.message
import email.errors
from email import quoprimime
def _encode_text(string, charset, cte, policy):
    lines = string.encode(charset).splitlines()
    linesep = policy.linesep.encode('ascii')

    def embedded_body(lines):
        return linesep.join(lines) + linesep

    def normal_body(lines):
        return b'\n'.join(lines) + b'\n'
    if cte is None:
        if max((len(x) for x in lines), default=0) <= policy.max_line_length:
            try:
                return ('7bit', normal_body(lines).decode('ascii'))
            except UnicodeDecodeError:
                pass
            if policy.cte_type == '8bit':
                return ('8bit', normal_body(lines).decode('ascii', 'surrogateescape'))
        sniff = embedded_body(lines[:10])
        sniff_qp = quoprimime.body_encode(sniff.decode('latin-1'), policy.max_line_length)
        sniff_base64 = binascii.b2a_base64(sniff)
        if len(sniff_qp) > len(sniff_base64):
            cte = 'base64'
        else:
            cte = 'quoted-printable'
            if len(lines) <= 10:
                return (cte, sniff_qp)
    if cte == '7bit':
        data = normal_body(lines).decode('ascii')
    elif cte == '8bit':
        data = normal_body(lines).decode('ascii', 'surrogateescape')
    elif cte == 'quoted-printable':
        data = quoprimime.body_encode(normal_body(lines).decode('latin-1'), policy.max_line_length)
    elif cte == 'base64':
        data = _encode_base64(embedded_body(lines), policy.max_line_length)
    else:
        raise ValueError('Unknown content transfer encoding {}'.format(cte))
    return (cte, data)