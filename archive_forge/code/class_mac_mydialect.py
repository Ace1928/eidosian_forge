import netaddr
class mac_mydialect(netaddr.mac_unix):
    word_fmt = '%.2x'