from netaddr.core import AddrFormatError, AddrConversionError
from netaddr.ip import IPRange, IPAddress, IPNetwork, iprange_to_cidrs
def _set_glob(self, ipglob):
    self._start, self._end = glob_to_iptuple(ipglob)
    self._glob = iprange_to_globs(self._start, self._end)[0]