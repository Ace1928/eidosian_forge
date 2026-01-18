import sys, re, curl, exceptions
from the command line first, then standard input.
class LinksysSession:
    months = 'Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec'
    WAN_CONNECT_AUTO = '1'
    WAN_CONNECT_STATIC = '2'
    WAN_CONNECT_PPOE = '3'
    WAN_CONNECT_RAS = '4'
    WAN_CONNECT_PPTP = '5'
    WAN_CONNECT_HEARTBEAT = '6'
    check_strings = {'': 'basic setup functions', 'Passwd.htm': 'For security reasons,', 'DHCP.html': 'You can configure the router to act as a DHCP', 'Log.html': 'There are some log settings and lists in this page.', 'Forward.htm': 'Port forwarding can be used to set up public services'}

    def __init__(self):
        self.actions = []
        self.host = 'http://192.168.1.1'
        self.verbosity = False
        self.pagecache = {}

    def set_verbosity(self, flag):
        self.verbosity = flag

    def cache_load(self, page):
        if page not in self.pagecache:
            fetch = curl.Curl(self.host)
            fetch.set_verbosity(self.verbosity)
            fetch.get(page)
            self.pagecache[page] = fetch.body()
            if fetch.answered('401'):
                raise LinksysError('authorization failure.', True)
            elif not fetch.answered(LinksysSession.check_strings[page]):
                del self.pagecache[page]
                raise LinksysError('check string for page %s missing!' % os.path.join(self.host, page), False)
            fetch.close()

    def cache_flush(self):
        self.pagecache = {}

    def screen_scrape(self, page, template):
        self.cache_load(page)
        match = re.compile(template).search(self.pagecache[page])
        if match:
            result = match.group(1)
        else:
            result = None
        return result

    def get_MAC_address(self, page, prefix):
        return self.screen_scrape('', prefix + ':[^M]*\\(MAC Address: *([^)]*)')

    def set_flag(self, page, flag, value):
        if value:
            self.actions.append(page, flag, '1')
        else:
            self.actions.append(page, flag, '0')

    def set_IP_address(self, page, cgi, role, ip):
        ind = 0
        for octet in ip.split('.'):
            self.actions.append(('', 'F1', role + repr(ind + 1), octet))
            ind += 1

    def get_firmware_version(self):
        return self.screen_scrape('', '>([0-9.v]*, (' + LinksysSession.months + ')[^<]*)<')

    def get_LAN_MAC(self):
        return self.get_MAC_address('', 'LAN IP Address')

    def get_Wireless_MAC(self):
        return self.get_MAC_address('', 'Wireless')

    def get_WAN_MAC(self):
        return self.get_MAC_address('', 'WAN Connection Type')

    def set_host_name(self, name):
        self.actions.append(('', 'hostName', name))

    def set_domain_name(self, name):
        self.actions.append(('', 'DomainName', name))

    def set_LAN_IP(self, ip):
        self.set_IP_address('', 'ipAddr', ip)

    def set_LAN_netmask(self, ip):
        if not ip.startswith('255.255.255.'):
            raise ValueError
        lastquad = ip.split('.')[-1]
        if lastquad not in ('0', '128', '192', '240', '252'):
            raise ValueError
        self.actions.append('', 'netMask', lastquad)

    def set_wireless(self, flag):
        self.set_flag('', 'wirelessStatus')

    def set_SSID(self, ssid):
        self.actions.append(('', 'wirelessESSID', ssid))

    def set_SSID_broadcast(self, flag):
        self.set_flag('', 'broadcastSSID')

    def set_channel(self, channel):
        self.actions.append(('', 'wirelessChannel', channel))

    def set_WEP(self, flag):
        self.set_flag('', 'WepType')

    def set_connection_type(self, type):
        self.actions.append(('', 'WANConnectionType', type))

    def set_WAN_IP(self, ip):
        self.set_IP_address('', 'aliasIP', ip)

    def set_WAN_netmask(self, ip):
        self.set_IP_address('', 'aliasMaskIP', ip)

    def set_WAN_gateway_address(self, ip):
        self.set_IP_address('', 'routerIP', ip)

    def set_DNS_server(self, index, ip):
        self.set_IP_address('', 'dns' + 'ABC'[index], ip)

    def set_password(self, str):
        self.actions.append('Passwd.htm', 'sysPasswd', str)
        self.actions.append('Passwd.htm', 'sysPasswdConfirm', str)

    def set_UPnP(self, flag):
        self.set_flag('Passwd.htm', 'UPnP_Work')

    def reset(self):
        self.actions.append('Passwd.htm', 'FactoryDefaults')

    def set_DHCP(self, flag):
        if flag:
            self.actions.append('DHCP.htm', 'dhcpStatus', 'Enable')
        else:
            self.actions.append('DHCP.htm', 'dhcpStatus', 'Disable')

    def set_DHCP_starting_IP(self, val):
        self.actions.append('DHCP.htm', 'dhcpS4', str(val))

    def set_DHCP_users(self, val):
        self.actions.append('DHCP.htm', 'dhcpLen', str(val))

    def set_DHCP_lease_time(self, val):
        self.actions.append('DHCP.htm', 'leaseTime', str(val))

    def set_DHCP_DNS_server(self, index, ip):
        self.set_IP_address('DHCP.htm', 'dns' + 'ABC'[index], ip)

    def set_logging(self, flag):
        if flag:
            self.actions.append('Log.htm', 'rLog', 'Enable')
        else:
            self.actions.append('Log.htm', 'rLog', 'Disable')

    def set_log_address(self, val):
        self.actions.append('DHCP.htm', 'trapAddr3', str(val))

    def configure(self):
        """Write configuration changes to the Linksys."""
        if self.actions:
            fields = []
            self.cache_flush()
            for page, field, value in self.actions:
                self.cache_load(page)
                if self.pagecache[page].find(field) == -1:
                    print_stderr('linksys: field %s not found where expected in page %s!' % (field, os.path.join(self.host, page)))
                    continue
                else:
                    fields.append((field, value))
            self.actions = []
            transaction = curl.Curl(self.host)
            transaction.set_verbosity(self.verbosity)
            transaction.get('Gozila.cgi', tuple(fields))
            transaction.close()