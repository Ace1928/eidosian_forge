import struct
from zope.interface import implementer
from twisted.internet import defer, interfaces as iinternet, protocol
from twisted.logger import Logger
from twisted.python.compat import iterbytes
from twisted.protocols import basic
from twisted.cred import credentials
class ITelnetProtocol(iinternet.IProtocol):

    def unhandledCommand(command, argument):
        """
        A command was received but not understood.

        @param command: the command received.
        @type command: L{str}, a single character.
        @param argument: the argument to the received command.
        @type argument: L{str}, a single character, or None if the command that
            was unhandled does not provide an argument.
        """

    def unhandledSubnegotiation(command, data):
        """
        A subnegotiation command was received but not understood.

        @param command: the command being subnegotiated. That is, the first
            byte after the SB command.
        @type command: L{str}, a single character.
        @param data: all other bytes of the subneogation. That is, all but the
            first bytes between SB and SE, with IAC un-escaping applied.
        @type data: L{bytes}, each a single character
        """

    def enableLocal(option):
        """
        Enable the given option locally.

        This should enable the given option on this side of the
        telnet connection and return True.  If False is returned,
        the option will be treated as still disabled and the peer
        will be notified.

        @param option: the option to be enabled.
        @type option: L{bytes}, a single character.
        """

    def enableRemote(option):
        """
        Indicate whether the peer should be allowed to enable this option.

        Returns True if the peer should be allowed to enable this option,
        False otherwise.

        @param option: the option to be enabled.
        @type option: L{bytes}, a single character.
        """

    def disableLocal(option):
        """
        Disable the given option locally.

        Unlike enableLocal, this method cannot fail.  The option must be
        disabled.

        @param option: the option to be disabled.
        @type option: L{bytes}, a single character.
        """

    def disableRemote(option):
        """
        Indicate that the peer has disabled this option.

        @param option: the option to be disabled.
        @type option: L{bytes}, a single character.
        """