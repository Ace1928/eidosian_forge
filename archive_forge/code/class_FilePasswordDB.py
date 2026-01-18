import os
from typing import Any, Dict, Optional, Tuple, Union
from zope.interface import Attribute, Interface, implementer
from twisted.cred import error
from twisted.cred.credentials import (
from twisted.internet import defer
from twisted.internet.defer import Deferred
from twisted.logger import Logger
from twisted.python import failure
@implementer(ICredentialsChecker)
class FilePasswordDB:
    """
    A file-based, text-based username/password database.

    Records in the datafile for this class are delimited by a particular
    string.  The username appears in a fixed field of the columns delimited
    by this string, as does the password.  Both fields are specifiable.  If
    the passwords are not stored plaintext, a hash function must be supplied
    to convert plaintext passwords to the form stored on disk and this
    CredentialsChecker will only be able to check L{IUsernamePassword}
    credentials.  If the passwords are stored plaintext,
    L{IUsernameHashedPassword} credentials will be checkable as well.
    """
    cache = False
    _credCache: Optional[Dict[bytes, bytes]] = None
    _cacheTimestamp: float = 0
    _log = Logger()

    def __init__(self, filename, delim=b':', usernameField=0, passwordField=1, caseSensitive=True, hash=None, cache=False):
        """
        @type filename: L{str}
        @param filename: The name of the file from which to read username and
        password information.

        @type delim: L{bytes}
        @param delim: The field delimiter used in the file.

        @type usernameField: L{int}
        @param usernameField: The index of the username after splitting a
        line on the delimiter.

        @type passwordField: L{int}
        @param passwordField: The index of the password after splitting a
        line on the delimiter.

        @type caseSensitive: L{bool}
        @param caseSensitive: If true, consider the case of the username when
        performing a lookup.  Ignore it otherwise.

        @type hash: Three-argument callable or L{None}
        @param hash: A function used to transform the plaintext password
        received over the network to a format suitable for comparison
        against the version stored on disk.  The arguments to the callable
        are the username, the network-supplied password, and the in-file
        version of the password.  If the return value compares equal to the
        version stored on disk, the credentials are accepted.

        @type cache: L{bool}
        @param cache: If true, maintain an in-memory cache of the
        contents of the password file.  On lookups, the mtime of the
        file will be checked, and the file will only be re-parsed if
        the mtime is newer than when the cache was generated.
        """
        self.filename = filename
        self.delim = delim
        self.ufield = usernameField
        self.pfield = passwordField
        self.caseSensitive = caseSensitive
        self.hash = hash
        self.cache = cache
        if self.hash is None:
            self.credentialInterfaces = (IUsernamePassword, IUsernameHashedPassword)
        else:
            self.credentialInterfaces = (IUsernamePassword,)

    def __getstate__(self):
        d = dict(vars(self))
        for k in ('_credCache', '_cacheTimestamp'):
            try:
                del d[k]
            except KeyError:
                pass
        return d

    def _cbPasswordMatch(self, matched, username):
        if matched:
            return username
        else:
            return failure.Failure(error.UnauthorizedLogin())

    def _loadCredentials(self):
        """
        Loads the credentials from the configured file.

        @return: An iterable of C{username, password} couples.
        @rtype: C{iterable}

        @raise UnauthorizedLogin: when failing to read the credentials from the
            file.
        """
        try:
            with open(self.filename, 'rb') as f:
                for line in f:
                    line = line.rstrip()
                    parts = line.split(self.delim)
                    if self.ufield >= len(parts) or self.pfield >= len(parts):
                        continue
                    if self.caseSensitive:
                        yield (parts[self.ufield], parts[self.pfield])
                    else:
                        yield (parts[self.ufield].lower(), parts[self.pfield])
        except OSError as e:
            self._log.error('Unable to load credentials db: {e!r}', e=e)
            raise error.UnauthorizedLogin()

    def getUser(self, username: bytes) -> Tuple[bytes, bytes]:
        """
        Look up the credentials for a username.

        @param username: The username to look up.
        @type username: L{bytes}

        @returns: Two-tuple of the canonicalicalized username (i.e. lowercase
        if the database is not case sensitive) and the associated password
        value, both L{bytes}.
        @rtype: L{tuple}

        @raises KeyError: When lookup of the username fails.
        """
        if not self.caseSensitive:
            username = username.lower()
        if self.cache:
            if self._credCache is None or os.path.getmtime(self.filename) > self._cacheTimestamp:
                self._cacheTimestamp = os.path.getmtime(self.filename)
                self._credCache = dict(self._loadCredentials())
            return (username, self._credCache[username])
        else:
            for u, p in self._loadCredentials():
                if u == username:
                    return (u, p)
            raise KeyError(username)

    def requestAvatarId(self, credentials: IUsernamePassword) -> Deferred[Union[bytes, Tuple[()]]]:
        try:
            u, p = self.getUser(credentials.username)
        except KeyError:
            return defer.fail(error.UnauthorizedLogin())
        else:
            up = IUsernamePassword(credentials, None)
            if self.hash:
                if up is not None:
                    h = self.hash(up.username, up.password, p)
                    if h == p:
                        return defer.succeed(u)
                return defer.fail(error.UnauthorizedLogin())
            else:
                return defer.maybeDeferred(credentials.checkPassword, p).addCallback(self._cbPasswordMatch, u)