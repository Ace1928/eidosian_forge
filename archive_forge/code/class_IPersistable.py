import os
import pickle
import sys
from zope.interface import Interface, implementer
from twisted.persisted import styles
from twisted.python import log, runtime
class IPersistable(Interface):
    """An object which can be saved in several formats to a file"""

    def setStyle(style):
        """Set desired format.

        @type style: string (one of 'pickle' or 'source')
        """

    def save(tag=None, filename=None, passphrase=None):
        """Save object to file.

        @type tag: string
        @type filename: string
        @type passphrase: string
        """