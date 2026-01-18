import sys
import dns._features
class _RegistryGetter:

    def __init__(self):
        self.info = DnsInfo()

    def _determine_split_char(self, entry):
        if entry.find(' ') >= 0:
            split_char = ' '
        elif entry.find(',') >= 0:
            split_char = ','
        else:
            split_char = ' '
        return split_char

    def _config_nameservers(self, nameservers):
        split_char = self._determine_split_char(nameservers)
        ns_list = nameservers.split(split_char)
        for ns in ns_list:
            if ns not in self.info.nameservers:
                self.info.nameservers.append(ns)

    def _config_search(self, search):
        split_char = self._determine_split_char(search)
        search_list = search.split(split_char)
        for s in search_list:
            s = _config_domain(s)
            if s not in self.info.search:
                self.info.search.append(s)

    def _config_fromkey(self, key, always_try_domain):
        try:
            servers, _ = winreg.QueryValueEx(key, 'NameServer')
        except WindowsError:
            servers = None
        if servers:
            self._config_nameservers(servers)
        if servers or always_try_domain:
            try:
                dom, _ = winreg.QueryValueEx(key, 'Domain')
                if dom:
                    self.info.domain = _config_domain(dom)
            except WindowsError:
                pass
        else:
            try:
                servers, _ = winreg.QueryValueEx(key, 'DhcpNameServer')
            except WindowsError:
                servers = None
            if servers:
                self._config_nameservers(servers)
                try:
                    dom, _ = winreg.QueryValueEx(key, 'DhcpDomain')
                    if dom:
                        self.info.domain = _config_domain(dom)
                except WindowsError:
                    pass
        try:
            search, _ = winreg.QueryValueEx(key, 'SearchList')
        except WindowsError:
            search = None
        if search is None:
            try:
                search, _ = winreg.QueryValueEx(key, 'DhcpSearchList')
            except WindowsError:
                search = None
        if search:
            self._config_search(search)

    def _is_nic_enabled(self, lm, guid):
        try:
            connection_key = winreg.OpenKey(lm, 'SYSTEM\\CurrentControlSet\\Control\\Network\\{4D36E972-E325-11CE-BFC1-08002BE10318}\\%s\\Connection' % guid)
            try:
                pnp_id, ttype = winreg.QueryValueEx(connection_key, 'PnpInstanceID')
                if ttype != winreg.REG_SZ:
                    raise ValueError
                device_key = winreg.OpenKey(lm, 'SYSTEM\\CurrentControlSet\\Enum\\%s' % pnp_id)
                try:
                    flags, ttype = winreg.QueryValueEx(device_key, 'ConfigFlags')
                    if ttype != winreg.REG_DWORD:
                        raise ValueError
                    return not flags & 1
                finally:
                    device_key.Close()
            finally:
                connection_key.Close()
        except Exception:
            return False

    def get(self):
        """Extract resolver configuration from the Windows registry."""
        lm = winreg.ConnectRegistry(None, winreg.HKEY_LOCAL_MACHINE)
        try:
            tcp_params = winreg.OpenKey(lm, 'SYSTEM\\CurrentControlSet\\Services\\Tcpip\\Parameters')
            try:
                self._config_fromkey(tcp_params, True)
            finally:
                tcp_params.Close()
            interfaces = winreg.OpenKey(lm, 'SYSTEM\\CurrentControlSet\\Services\\Tcpip\\Parameters\\Interfaces')
            try:
                i = 0
                while True:
                    try:
                        guid = winreg.EnumKey(interfaces, i)
                        i += 1
                        key = winreg.OpenKey(interfaces, guid)
                        try:
                            if not self._is_nic_enabled(lm, guid):
                                continue
                            self._config_fromkey(key, False)
                        finally:
                            key.Close()
                    except EnvironmentError:
                        break
            finally:
                interfaces.Close()
        finally:
            lm.Close()
        return self.info