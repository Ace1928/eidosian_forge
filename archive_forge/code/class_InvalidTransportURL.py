import logging
from debtcollector import removals
from oslo_config import cfg
from stevedore import driver
from urllib import parse
from oslo_messaging import exceptions
class InvalidTransportURL(exceptions.MessagingException):
    """Raised if transport URL is invalid."""

    def __init__(self, url, msg):
        super(InvalidTransportURL, self).__init__(msg)
        self.url = url