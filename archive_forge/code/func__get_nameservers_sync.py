import socket
import time
from urllib.parse import urlparse
import dns.asyncbackend
import dns.inet
import dns.name
import dns.nameserver
import dns.query
import dns.rdtypes.svcbbase
def _get_nameservers_sync(answer, lifetime):
    """Return a list of TLS-validated resolver nameservers extracted from an SVCB
    answer."""
    nameservers = []
    infos = _extract_nameservers_from_svcb(answer)
    for info in infos:
        try:
            if info.ddr_tls_check_sync(lifetime):
                nameservers.extend(info.nameservers)
        except Exception:
            pass
    return nameservers