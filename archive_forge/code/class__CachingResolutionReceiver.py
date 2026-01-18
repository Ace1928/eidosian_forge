from typing import Any
from twisted.internet import defer
from twisted.internet.base import ThreadedResolver
from twisted.internet.interfaces import (
from zope.interface.declarations import implementer, provider
from scrapy.utils.datatypes import LocalCache
@provider(IResolutionReceiver)
class _CachingResolutionReceiver:

    def __init__(self, resolutionReceiver, hostName):
        self.resolutionReceiver = resolutionReceiver
        self.hostName = hostName
        self.addresses = []

    def resolutionBegan(self, resolution):
        self.resolutionReceiver.resolutionBegan(resolution)
        self.resolution = resolution

    def addressResolved(self, address):
        self.resolutionReceiver.addressResolved(address)
        self.addresses.append(address)

    def resolutionComplete(self):
        self.resolutionReceiver.resolutionComplete()
        if self.addresses:
            dnscache[self.hostName] = self.addresses