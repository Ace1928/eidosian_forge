import logging
from debtcollector import removals
from oslo_config import cfg
from stevedore import driver
from urllib import parse
from oslo_messaging import exceptions
class RPCTransport(Transport):
    """Transport object for RPC."""

    def __init__(self, driver):
        super(RPCTransport, self).__init__(driver)