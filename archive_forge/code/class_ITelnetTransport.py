import struct
from zope.interface import implementer
from twisted.internet import defer, interfaces as iinternet, protocol
from twisted.logger import Logger
from twisted.python.compat import iterbytes
from twisted.protocols import basic
from twisted.cred import credentials
class ITelnetTransport(iinternet.ITransport):

    def do(option):
        """
        Indicate a desire for the peer to begin performing the given option.

        Returns a Deferred that fires with True when the peer begins performing
        the option, or fails with L{OptionRefused} when the peer refuses to
        perform it.  If the peer is already performing the given option, the
        Deferred will fail with L{AlreadyEnabled}.  If a negotiation regarding
        this option is already in progress, the Deferred will fail with
        L{AlreadyNegotiating}.

        Note: It is currently possible that this Deferred will never fire,
        if the peer never responds, or if the peer believes the option to
        already be enabled.
        """

    def dont(option):
        """
        Indicate a desire for the peer to cease performing the given option.

        Returns a Deferred that fires with True when the peer ceases performing
        the option.  If the peer is not performing the given option, the
        Deferred will fail with L{AlreadyDisabled}.  If negotiation regarding
        this option is already in progress, the Deferred will fail with
        L{AlreadyNegotiating}.

        Note: It is currently possible that this Deferred will never fire,
        if the peer never responds, or if the peer believes the option to
        already be disabled.
        """

    def will(option):
        """
        Indicate our willingness to begin performing this option locally.

        Returns a Deferred that fires with True when the peer agrees to allow us
        to begin performing this option, or fails with L{OptionRefused} if the
        peer refuses to allow us to begin performing it.  If the option is
        already enabled locally, the Deferred will fail with L{AlreadyEnabled}.
        If negotiation regarding this option is already in progress, the
        Deferred will fail with L{AlreadyNegotiating}.

        Note: It is currently possible that this Deferred will never fire,
        if the peer never responds, or if the peer believes the option to
        already be enabled.
        """

    def wont(option):
        """
        Indicate that we will stop performing the given option.

        Returns a Deferred that fires with True when the peer acknowledges
        we have stopped performing this option.  If the option is already
        disabled locally, the Deferred will fail with L{AlreadyDisabled}.
        If negotiation regarding this option is already in progress,
        the Deferred will fail with L{AlreadyNegotiating}.

        Note: It is currently possible that this Deferred will never fire,
        if the peer never responds, or if the peer believes the option to
        already be disabled.
        """

    def requestNegotiation(about, data):
        """
        Send a subnegotiation request.

        @param about: A byte indicating the feature being negotiated.
        @param data: Any number of L{bytes} containing specific information
        about the negotiation being requested.  No values in this string
        need to be escaped, as this function will escape any value which
        requires it.
        """