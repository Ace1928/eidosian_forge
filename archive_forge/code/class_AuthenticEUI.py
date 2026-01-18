import random
import socket
import netaddr
from neutron_lib import constants
class AuthenticEUI(_AuthenticBase, netaddr.EUI):
    """AuthenticEUI class

    This class retains the format of the MAC address string passed during
    initialization.

    This is useful when we want to make sure that we retain the format passed
    by a user through API.
    """