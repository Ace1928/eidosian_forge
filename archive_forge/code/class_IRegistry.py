import socket
import time
import warnings
from collections import OrderedDict
from typing import Dict, List
from zope.interface import Interface, implementer
from twisted import cred
from twisted.internet import defer, protocol, reactor
from twisted.protocols import basic
from twisted.python import log
class IRegistry(Interface):
    """
    Allows registration of logical->physical URL mapping.
    """

    def registerAddress(domainURL, logicalURL, physicalURL):
        """
        Register the physical address of a logical URL.

        @return: Deferred of C{Registration} or failure with RegistrationError.
        """

    def unregisterAddress(domainURL, logicalURL, physicalURL):
        """
        Unregister the physical address of a logical URL.

        @return: Deferred of C{Registration} or failure with RegistrationError.
        """

    def getRegistrationInfo(logicalURL):
        """
        Get registration info for logical URL.

        @return: Deferred of C{Registration} object or failure of LookupError.
        """