import sys, re, curl, exceptions
from the command line first, then standard input.
class LinksysInterpreter(cmd.Cmd):
    """Interpret commands to perform LinkSys programming actions."""

    def __init__(self):
        cmd.Cmd.__init__(self)
        self.session = LinksysSession()
        if os.isatty(0):
            print("Type ? or `help' for help.")
            self.prompt = self.session.host + ': '
        else:
            self.prompt = ''
            print('Bar1')

    def flag_command(self, func, line):
        if line.strip() in ('on', 'enable', 'yes'):
            func(True)
        elif line.strip() in ('off', 'disable', 'no'):
            func(False)
        else:
            print_stderr('linksys: unknown switch value')
        return 0

    def do_connect(self, line):
        newhost = line.strip()
        if newhost:
            self.session.host = newhost
            self.session.cache_flush()
            self.prompt = self.session.host + ': '
        else:
            print(self.session.host)
        return 0

    def help_connect(self):
        print('Usage: connect [<hostname-or-IP>]')
        print('Connect to a Linksys by name or IP address.')
        print('If no argument is given, print the current host.')

    def do_status(self, line):
        self.session.cache_load('')
        if '' in self.session.pagecache:
            print('Firmware:', self.session.get_firmware_version())
            print('LAN MAC:', self.session.get_LAN_MAC())
            print('Wireless MAC:', self.session.get_Wireless_MAC())
            print('WAN MAC:', self.session.get_WAN_MAC())
            print('.')
        return 0

    def help_status(self):
        print('Usage: status')
        print('The status command shows the status of the Linksys.')
        print('It is mainly useful as a sanity check to make sure')
        print('the box is responding correctly.')

    def do_verbose(self, line):
        self.flag_command(self.session.set_verbosity, line)

    def help_verbose(self):
        print('Usage: verbose {on|off|enable|disable|yes|no}')
        print('Enables display of HTTP requests.')

    def do_host(self, line):
        self.session.set_host_name(line)
        return 0

    def help_host(self):
        print('Usage: host <hostname>')
        print('Sets the Host field to be queried by the ISP.')

    def do_domain(self, line):
        print('Usage: host <domainname>')
        self.session.set_domain_name(line)
        return 0

    def help_domain(self):
        print('Sets the Domain field to be queried by the ISP.')

    def do_lan_address(self, line):
        self.session.set_LAN_IP(line)
        return 0

    def help_lan_address(self):
        print('Usage: lan_address <ip-address>')
        print('Sets the LAN IP address.')

    def do_lan_netmask(self, line):
        self.session.set_LAN_netmask(line)
        return 0

    def help_lan_netmask(self):
        print('Usage: lan_netmask <ip-mask>')
        print('Sets the LAN subnetwork mask.')

    def do_wireless(self, line):
        self.flag_command(self.session.set_wireless, line)
        return 0

    def help_wireless(self):
        print('Usage: wireless {on|off|enable|disable|yes|no}')
        print('Switch to enable or disable wireless features.')

    def do_ssid(self, line):
        self.session.set_SSID(line)
        return 0

    def help_ssid(self):
        print('Usage: ssid <string>')
        print('Sets the SSID used to control wireless access.')

    def do_ssid_broadcast(self, line):
        self.flag_command(self.session.set_SSID_broadcast, line)
        return 0

    def help_ssid_broadcast(self):
        print('Usage: ssid_broadcast {on|off|enable|disable|yes|no}')
        print('Switch to enable or disable SSID broadcast.')

    def do_channel(self, line):
        self.session.set_channel(line)
        return 0

    def help_channel(self):
        print('Usage: channel <number>')
        print('Sets the wireless channel.')

    def do_wep(self, line):
        self.flag_command(self.session.set_WEP, line)
        return 0

    def help_wep(self):
        print('Usage: wep {on|off|enable|disable|yes|no}')
        print('Switch to enable or disable WEP security.')

    def do_wan_type(self, line):
        try:
            type = eval('LinksysSession.WAN_CONNECT_' + line.strip().upper())
            self.session.set_connection_type(type)
        except ValueError:
            print_stderr('linksys: unknown connection type.')
        return 0

    def help_wan_type(self):
        print('Usage: wan_type {auto|static|ppoe|ras|pptp|heartbeat}')
        print('Set the WAN connection type.')

    def do_wan_address(self, line):
        self.session.set_WAN_IP(line)
        return 0

    def help_wan_address(self):
        print('Usage: wan_address <ip-address>')
        print('Sets the WAN IP address.')

    def do_wan_netmask(self, line):
        self.session.set_WAN_netmask(line)
        return 0

    def help_wan_netmask(self):
        print('Usage: wan_netmask <ip-mask>')
        print('Sets the WAN subnetwork mask.')

    def do_wan_gateway(self, line):
        self.session.set_WAN_gateway(line)
        return 0

    def help_wan_gateway(self):
        print('Usage: wan_gateway <ip-address>')
        print('Sets the LAN subnetwork mask.')

    def do_dns(self, line):
        index, address = line.split()
        if index in ('1', '2', '3'):
            self.session.set_DNS_server(eval(index), address)
        else:
            print_stderr('linksys: server index out of bounds.')
        return 0

    def help_dns(self):
        print('Usage: dns {1|2|3} <ip-mask>')
        print('Sets a primary, secondary, or tertiary DNS server address.')

    def do_password(self, line):
        self.session.set_password(line)
        return 0

    def help_password(self):
        print('Usage: password <string>')
        print('Sets the router password.')

    def do_upnp(self, line):
        self.flag_command(self.session.set_UPnP, line)
        return 0

    def help_upnp(self):
        print('Usage: upnp {on|off|enable|disable|yes|no}')
        print('Switch to enable or disable Universal Plug and Play.')

    def do_reset(self, line):
        self.session.reset()

    def help_reset(self):
        print('Usage: reset')
        print('Reset Linksys settings to factory defaults.')

    def do_dhcp(self, line):
        self.flag_command(self.session.set_DHCP, line)

    def help_dhcp(self):
        print('Usage: dhcp {on|off|enable|disable|yes|no}')
        print('Switch to enable or disable DHCP features.')

    def do_dhcp_start(self, line):
        self.session.set_DHCP_starting_IP(line)

    def help_dhcp_start(self):
        print('Usage: dhcp_start <number>')
        print('Set the start address of the DHCP pool.')

    def do_dhcp_users(self, line):
        self.session.set_DHCP_users(line)

    def help_dhcp_users(self):
        print('Usage: dhcp_users <number>')
        print('Set number of address slots to allocate in the DHCP pool.')

    def do_dhcp_lease(self, line):
        self.session.set_DHCP_lease(line)

    def help_dhcp_lease(self):
        print('Usage: dhcp_lease <number>')
        print('Set number of address slots to allocate in the DHCP pool.')

    def do_dhcp_dns(self, line):
        index, address = line.split()
        if index in ('1', '2', '3'):
            self.session.set_DHCP_DNS_server(eval(index), address)
        else:
            print_stderr('linksys: server index out of bounds.')
        return 0

    def help_dhcp_dns(self):
        print('Usage: dhcp_dns {1|2|3} <ip-mask>')
        print('Sets primary, secondary, or tertiary DNS server address.')

    def do_logging(self, line):
        self.flag_command(self.session.set_logging, line)

    def help_logging(self):
        print('Usage: logging {on|off|enable|disable|yes|no}')
        print('Switch to enable or disable session logging.')

    def do_log_address(self, line):
        self.session.set_Log_address(line)

    def help_log_address(self):
        print('Usage: log_address <number>')
        print('Set the last quad of the address to which to log.')

    def do_configure(self, line):
        self.session.configure()
        return 0

    def help_configure(self):
        print('Usage: configure')
        print('Writes the configuration to the Linksys.')

    def do_cache(self, line):
        print(self.session.pagecache)

    def help_cache(self):
        print('Usage: cache')
        print('Display the page cache.')

    def do_quit(self, line):
        return 1

    def help_quit(self, line):
        print('The quit command ends your linksys session without')
        print('writing configuration changes to the Linksys.')

    def do_EOF(self, line):
        print('')
        self.session.configure()
        return 1

    def help_EOF(self):
        print('The EOF command writes the configuration to the linksys')
        print('and ends your session.')

    def default(self, line):
        """Pass the command through to be executed by the shell."""
        os.system(line)
        return 0

    def help_help(self):
        print('On-line help is available through this command.')
        print('? is a convenience alias for help.')

    def help_introduction(self):
        print('\nThis program supports changing the settings on Linksys blue-box routers.  This\ncapability may come in handy when they freeze up and have to be reset.  Though\nit can be used interactively (and will command-prompt when standard input is a\nterminal) it is really designed to be used in batch mode. Commands are taken\nfrom the command line first, then standard input.\n\nBy default, it is assumed that the Linksys is at http://192.168.1.1, the\ndefault LAN address.  You can connect to a different address or IP with the\n\'connect\' command.  Note that your .netrc must contain correct user/password\ncredentials for the router.  The entry corresponding to the defaults is:\n\nmachine 192.168.1.1\n    login ""\n    password admin\n\nMost commands queue up changes but don\'t actually send them to the Linksys.\nYou can force pending changes to be written with \'configure\'.  Otherwise, they\nwill be shipped to the Linksys at the end of session (e.g.  when the program\nrunning in batch mode encounters end-of-file or you type a control-D).  If you\nend the session with `quit\', pending changes will be discarded.\n\nFor more help, read the topics \'wan\', \'lan\', and \'wireless\'.')

    def help_lan(self):
        print("The `lan_address' and `lan_netmask' commands let you set the IP location of\nthe Linksys on your LAN, or inside.  Normally you'll want to leave these\nuntouched.")

    def help_wan(self):
        print("The WAN commands become significant if you are using the BEFSR41 or any of\nthe other Linksys boxes designed as DSL or cable-modem gateways.  You will\nneed to use `wan_type' to declare how you expect to get your address.\n\nIf your ISP has issued you a static address, you'll need to use the\n`wan_address', `wan_netmask', and `wan_gateway' commands to set the address\nof the router as seen from the WAN, the outside. In this case you will also\nneed to use the `dns' command to declare which remote servers your DNS\nrequests should be forwarded to.\n\nSome ISPs may require you to set host and domain for use with dynamic-address\nallocation.")

    def help_wireless_desc(self):
        print('The channel, ssid, ssid_broadcast, wep, and wireless commands control\nwireless routing.')

    def help_switches(self):
        print("Switches may be turned on with 'on', 'enable', or 'yes'.")
        print("Switches may be turned off with 'off', 'disable', or 'no'.")
        print('Switch commands include: wireless, ssid_broadcast.')

    def help_addresses(self):
        print('An address argument must be a valid IP address;')
        print('four decimal numbers separated by dots, each ')
        print('between 0 and 255.')

    def emptyline(self):
        pass