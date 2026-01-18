import socket
import sys
import time
import random
import dns.exception
import dns.flags
import dns.ipv4
import dns.ipv6
import dns.message
import dns.name
import dns.query
import dns.rcode
import dns.rdataclass
import dns.rdatatype
import dns.reversename
import dns.tsig
from ._compat import xrange, string_types
def _win32_is_nic_enabled(self, lm, guid, interface_key):
    try:
        connection_key = _winreg.OpenKey(lm, 'SYSTEM\\CurrentControlSet\\Control\\Network\\{4D36E972-E325-11CE-BFC1-08002BE10318}\\%s\\Connection' % guid)
        try:
            pnp_id, ttype = _winreg.QueryValueEx(connection_key, 'PnpInstanceID')
            if ttype != _winreg.REG_SZ:
                raise ValueError
            device_key = _winreg.OpenKey(lm, 'SYSTEM\\CurrentControlSet\\Enum\\%s' % pnp_id)
            try:
                flags, ttype = _winreg.QueryValueEx(device_key, 'ConfigFlags')
                if ttype != _winreg.REG_DWORD:
                    raise ValueError
                return not flags & 1
            finally:
                device_key.Close()
        finally:
            connection_key.Close()
    except (EnvironmentError, ValueError):
        try:
            nte, ttype = _winreg.QueryValueEx(interface_key, 'NTEContextList')
            return nte is not None
        except WindowsError:
            return False