import re
import sys
def _dnsname_match(dn, hostname, max_wildcards=1):
    """Matching according to RFC 6125, section 6.4.3

    http://tools.ietf.org/html/rfc6125#section-6.4.3
    """
    pats = []
    if not dn:
        return False
    parts = dn.split('.')
    leftmost = parts[0]
    remainder = parts[1:]
    wildcards = leftmost.count('*')
    if wildcards > max_wildcards:
        raise CertificateError('too many wildcards in certificate DNS name: ' + repr(dn))
    if not wildcards:
        return dn.lower() == hostname.lower()
    if leftmost == '*':
        pats.append('[^.]+')
    elif leftmost.startswith('xn--') or hostname.startswith('xn--'):
        pats.append(re.escape(leftmost))
    else:
        pats.append(re.escape(leftmost).replace('\\*', '[^.]*'))
    for frag in remainder:
        pats.append(re.escape(frag))
    pat = re.compile('\\A' + '\\.'.join(pats) + '\\Z', re.IGNORECASE)
    return pat.match(hostname)