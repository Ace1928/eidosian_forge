from zope.interface import Attribute, Interface
class IKnownHostEntry(Interface):
    """
    A L{IKnownHostEntry} is an entry in an OpenSSH-formatted C{known_hosts}
    file.

    @since: 8.2
    """

    def matchesKey(key):
        """
        Return True if this entry matches the given Key object, False
        otherwise.

        @param key: The key object to match against.
        @type key: L{twisted.conch.ssh.keys.Key}
        """

    def matchesHost(hostname):
        """
        Return True if this entry matches the given hostname, False otherwise.

        Note that this does no name resolution; if you want to match an IP
        address, you have to resolve it yourself, and pass it in as a dotted
        quad string.

        @param hostname: The hostname to match against.
        @type hostname: L{str}
        """

    def toString():
        """

        @return: a serialized string representation of this entry, suitable for
        inclusion in a known_hosts file.  (Newline not included.)

        @rtype: L{str}
        """