import functools
class _IPv4Constants:
    _linklocal_network = IPv4Network('169.254.0.0/16')
    _loopback_network = IPv4Network('127.0.0.0/8')
    _multicast_network = IPv4Network('224.0.0.0/4')
    _public_network = IPv4Network('100.64.0.0/10')
    _private_networks = [IPv4Network('0.0.0.0/8'), IPv4Network('10.0.0.0/8'), IPv4Network('127.0.0.0/8'), IPv4Network('169.254.0.0/16'), IPv4Network('172.16.0.0/12'), IPv4Network('192.0.0.0/29'), IPv4Network('192.0.0.170/31'), IPv4Network('192.0.2.0/24'), IPv4Network('192.168.0.0/16'), IPv4Network('198.18.0.0/15'), IPv4Network('198.51.100.0/24'), IPv4Network('203.0.113.0/24'), IPv4Network('240.0.0.0/4'), IPv4Network('255.255.255.255/32')]
    _reserved_network = IPv4Network('240.0.0.0/4')
    _unspecified_address = IPv4Address('0.0.0.0')