import hmac
import sys
from binascii import Error as DecodeError, a2b_base64, b2a_base64
from contextlib import closing
from hashlib import sha1
from zope.interface import implementer
from twisted.conch.error import HostKeyChanged, InvalidEntry, UserRejectedKey
from twisted.conch.interfaces import IKnownHostEntry
from twisted.conch.ssh.keys import BadKeyError, FingerprintFormats, Key
from twisted.internet import defer
from twisted.logger import Logger
from twisted.python.compat import nativeString
from twisted.python.randbytes import secureRandom
from twisted.python.util import FancyEqMixin
class KnownHostsFile:
    """
    A structured representation of an OpenSSH-format ~/.ssh/known_hosts file.

    @ivar _added: A list of L{IKnownHostEntry} providers which have been added
        to this instance in memory but not yet saved.

    @ivar _clobber: A flag indicating whether the current contents of the save
        path will be disregarded and potentially overwritten or not.  If
        C{True}, this will be done.  If C{False}, entries in the save path will
        be read and new entries will be saved by appending rather than
        overwriting.
    @type _clobber: L{bool}

    @ivar _savePath: See C{savePath} parameter of L{__init__}.
    """

    def __init__(self, savePath):
        """
        Create a new, empty KnownHostsFile.

        Unless you want to erase the current contents of C{savePath}, you want
        to use L{KnownHostsFile.fromPath} instead.

        @param savePath: The L{FilePath} to which to save new entries.
        @type savePath: L{FilePath}
        """
        self._added = []
        self._savePath = savePath
        self._clobber = True

    @property
    def savePath(self):
        """
        @see: C{savePath} parameter of L{__init__}
        """
        return self._savePath

    def iterentries(self):
        """
        Iterate over the host entries in this file.

        @return: An iterable the elements of which provide L{IKnownHostEntry}.
            There is an element for each entry in the file as well as an element
            for each added but not yet saved entry.
        @rtype: iterable of L{IKnownHostEntry} providers
        """
        for entry in self._added:
            yield entry
        if self._clobber:
            return
        try:
            fp = self._savePath.open()
        except OSError:
            return
        with fp:
            for line in fp:
                try:
                    if line.startswith(HashedEntry.MAGIC):
                        entry = HashedEntry.fromString(line)
                    else:
                        entry = PlainEntry.fromString(line)
                except (DecodeError, InvalidEntry, BadKeyError):
                    entry = UnparsedEntry(line)
                yield entry

    def hasHostKey(self, hostname, key):
        """
        Check for an entry with matching hostname and key.

        @param hostname: A hostname or IP address literal to check for.
        @type hostname: L{bytes}

        @param key: The public key to check for.
        @type key: L{Key}

        @return: C{True} if the given hostname and key are present in this file,
            C{False} if they are not.
        @rtype: L{bool}

        @raise HostKeyChanged: if the host key found for the given hostname
            does not match the given key.
        """
        for lineidx, entry in enumerate(self.iterentries(), -len(self._added)):
            if entry.matchesHost(hostname) and entry.keyType == key.sshType():
                if entry.matchesKey(key):
                    return True
                else:
                    if lineidx < 0:
                        line = None
                        path = None
                    else:
                        line = lineidx + 1
                        path = self._savePath
                    raise HostKeyChanged(entry, path, line)
        return False

    def verifyHostKey(self, ui, hostname, ip, key):
        """
        Verify the given host key for the given IP and host, asking for
        confirmation from, and notifying, the given UI about changes to this
        file.

        @param ui: The user interface to request an IP address from.

        @param hostname: The hostname that the user requested to connect to.

        @param ip: The string representation of the IP address that is actually
        being connected to.

        @param key: The public key of the server.

        @return: a L{Deferred} that fires with True when the key has been
            verified, or fires with an errback when the key either cannot be
            verified or has changed.
        @rtype: L{Deferred}
        """
        hhk = defer.execute(self.hasHostKey, hostname, key)

        def gotHasKey(result):
            if result:
                if not self.hasHostKey(ip, key):
                    ui.warn("Warning: Permanently added the %s host key for IP address '%s' to the list of known hosts." % (key.type(), nativeString(ip)))
                    self.addHostKey(ip, key)
                    self.save()
                return result
            else:

                def promptResponse(response):
                    if response:
                        self.addHostKey(hostname, key)
                        self.addHostKey(ip, key)
                        self.save()
                        return response
                    else:
                        raise UserRejectedKey()
                keytype = key.type()
                if keytype == 'EC':
                    keytype = 'ECDSA'
                prompt = "The authenticity of host '%s (%s)' can't be established.\n%s key fingerprint is SHA256:%s.\nAre you sure you want to continue connecting (yes/no)? " % (nativeString(hostname), nativeString(ip), keytype, key.fingerprint(format=FingerprintFormats.SHA256_BASE64))
                proceed = ui.prompt(prompt.encode(sys.getdefaultencoding()))
                return proceed.addCallback(promptResponse)
        return hhk.addCallback(gotHasKey)

    def addHostKey(self, hostname, key):
        """
        Add a new L{HashedEntry} to the key database.

        Note that you still need to call L{KnownHostsFile.save} if you wish
        these changes to be persisted.

        @param hostname: A hostname or IP address literal to associate with the
            new entry.
        @type hostname: L{bytes}

        @param key: The public key to associate with the new entry.
        @type key: L{Key}

        @return: The L{HashedEntry} that was added.
        @rtype: L{HashedEntry}
        """
        salt = secureRandom(20)
        keyType = key.sshType()
        entry = HashedEntry(salt, _hmacedString(salt, hostname), keyType, key, None)
        self._added.append(entry)
        return entry

    def save(self):
        """
        Save this L{KnownHostsFile} to the path it was loaded from.
        """
        p = self._savePath.parent()
        if not p.isdir():
            p.makedirs()
        if self._clobber:
            mode = 'wb'
        else:
            mode = 'ab'
        with self._savePath.open(mode) as hostsFileObj:
            if self._added:
                hostsFileObj.write(b'\n'.join([entry.toString() for entry in self._added]) + b'\n')
                self._added = []
        self._clobber = False

    @classmethod
    def fromPath(cls, path):
        """
        Create a new L{KnownHostsFile}, potentially reading existing known
        hosts information from the given file.

        @param path: A path object to use for both reading contents from and
            later saving to.  If no file exists at this path, it is not an
            error; a L{KnownHostsFile} with no entries is returned.
        @type path: L{FilePath}

        @return: A L{KnownHostsFile} initialized with entries from C{path}.
        @rtype: L{KnownHostsFile}
        """
        knownHosts = cls(path)
        knownHosts._clobber = False
        return knownHosts