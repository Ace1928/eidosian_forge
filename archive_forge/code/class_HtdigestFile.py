from __future__ import with_statement
import logging; log = logging.getLogger(__name__)
import os
from warnings import warn
from passlib import exc, registry
from passlib.context import CryptContext
from passlib.exc import ExpectedStringError
from passlib.hash import htdigest
from passlib.utils import render_bytes, to_bytes, is_ascii_codec
from passlib.utils.decor import deprecated_method
from passlib.utils.compat import join_bytes, unicode, BytesIO, PY3
class HtdigestFile(_CommonFile):
    """class for reading & writing Htdigest files.

    The class constructor accepts the following arguments:

    :type path: filepath
    :param path:

        Specifies path to htdigest file, use to implicitly load from and save to.

        This class has two modes of operation:

        1. It can be "bound" to a local file by passing a ``path`` to the class
           constructor. In this case it will load the contents of the file when
           created, and the :meth:`load` and :meth:`save` methods will automatically
           load from and save to that file if they are called without arguments.

        2. Alternately, it can exist as an independant object, in which case
           :meth:`load` and :meth:`save` will require an explicit path to be
           provided whenever they are called. As well, ``autosave`` behavior
           will not be available.

           This feature is new in Passlib 1.6, and is the default if no
           ``path`` value is provided to the constructor.

        This is also exposed as a readonly instance attribute.

    :type default_realm: str
    :param default_realm:

        If ``default_realm`` is set, all the :class:`HtdigestFile`
        methods that require a realm will use this value if one is not
        provided explicitly. If unset, they will raise an error stating
        that an explicit realm is required.

        This is also exposed as a writeable instance attribute.

        .. versionadded:: 1.6

    :type new: bool
    :param new:

        Normally, if *path* is specified, :class:`HtdigestFile` will
        immediately load the contents of the file. However, when creating
        a new htpasswd file, applications can set ``new=True`` so that
        the existing file (if any) will not be loaded.

        .. versionadded:: 1.6
            This feature was previously enabled by setting ``autoload=False``.
            That alias has been deprecated, and will be removed in Passlib 1.8

    :type autosave: bool
    :param autosave:

        Normally, any changes made to an :class:`HtdigestFile` instance
        will not be saved until :meth:`save` is explicitly called. However,
        if ``autosave=True`` is specified, any changes made will be
        saved to disk immediately (assuming *path* has been set).

        This is also exposed as a writeable instance attribute.

    :type encoding: str
    :param encoding:

        Optionally specify character encoding used to read/write file
        and hash passwords. Defaults to ``utf-8``, though ``latin-1``
        is the only other commonly encountered encoding.

        This is also exposed as a readonly instance attribute.

    :param autoload:
        Set to ``False`` to prevent the constructor from automatically
        loaded the file from disk.

        .. deprecated:: 1.6
            This has been replaced by the *new* keyword.
            Instead of setting ``autoload=False``, you should use
            ``new=True``. Support for this keyword will be removed
            in Passlib 1.8.

    Loading & Saving
    ================
    .. automethod:: load
    .. automethod:: load_if_changed
    .. automethod:: load_string
    .. automethod:: save
    .. automethod:: to_string

    Inspection
    ==========
    .. automethod:: realms
    .. automethod:: users
    .. automethod:: check_password(user[, realm], password)
    .. automethod:: get_hash

    Modification
    ============
    .. automethod:: set_password(user[, realm], password)
    .. automethod:: delete
    .. automethod:: delete_realm

    Alternate Constructors
    ======================
    .. automethod:: from_string

    Attributes
    ==========
    .. attribute:: default_realm

        The default realm that will be used if one is not provided
        to methods that require it. By default this is ``None``,
        in which case an explicit realm must be provided for every
        method call. Can be written to.

    .. attribute:: path

        Path to local file that will be used as the default
        for all :meth:`load` and :meth:`save` operations.
        May be written to, initialized by the *path* constructor keyword.

    .. attribute:: autosave

        Writeable flag indicating whether changes will be automatically
        written to *path*.

    Errors
    ======
    :raises ValueError:
        All of the methods in this class will raise a :exc:`ValueError` if
        any user name or realm contains a forbidden character (one of ``:\\r\\n\\t\\x00``),
        or is longer than 255 characters.
    """
    default_realm = None

    def __init__(self, path=None, default_realm=None, **kwds):
        self.default_realm = default_realm
        super(HtdigestFile, self).__init__(path, **kwds)

    def _parse_record(self, record, lineno):
        result = record.rstrip().split(_BCOLON)
        if len(result) != 3:
            raise ValueError('malformed htdigest file (error reading line %d)' % lineno)
        user, realm, hash = result
        return ((user, realm), hash)

    def _render_record(self, key, hash):
        user, realm = key
        return render_bytes('%s:%s:%s\n', user, realm, hash)

    def _require_realm(self, realm):
        if realm is None:
            realm = self.default_realm
            if realm is None:
                raise TypeError('you must specify a realm explicitly, or set the default_realm attribute')
        return realm

    def _encode_realm(self, realm):
        realm = self._require_realm(realm)
        return self._encode_field(realm, 'realm')

    def _encode_key(self, user, realm):
        return (self._encode_user(user), self._encode_realm(realm))

    def realms(self):
        """Return list of all realms in database"""
        realms = set((key[1] for key in self._records))
        return [self._decode_field(realm) for realm in realms]

    def users(self, realm=None):
        """Return list of all users in specified realm.

        * uses ``self.default_realm`` if no realm explicitly provided.
        * returns empty list if realm not found.
        """
        realm = self._encode_realm(realm)
        return [self._decode_field(key[0]) for key in self._records if key[1] == realm]

    def set_password(self, user, realm=None, password=_UNSET):
        """Set password for user; adds user & realm if needed.

        If ``self.default_realm`` has been set, this may be called
        with the syntax ``set_password(user, password)``,
        otherwise it must be called with all three arguments:
        ``set_password(user, realm, password)``.

        :returns:
            * ``True`` if existing user was updated
            * ``False`` if user account added.
        """
        if password is _UNSET:
            realm, password = (None, realm)
        realm = self._require_realm(realm)
        hash = htdigest.hash(password, user, realm, encoding=self.encoding)
        return self.set_hash(user, realm, hash)

    @deprecated_method(deprecated='1.6', removed='1.8', replacement='set_password')
    def update(self, user, realm, password):
        """set password for user"""
        return self.set_password(user, realm, password)

    def get_hash(self, user, realm=None):
        """Return :class:`~passlib.hash.htdigest` hash stored for user.

        * uses ``self.default_realm`` if no realm explicitly provided.
        * returns ``None`` if user or realm not found.

        .. versionchanged:: 1.6
            This method was previously named ``find``, it was renamed
            for clarity. The old name is deprecated, and will be removed
            in Passlib 1.8.
        """
        key = self._encode_key(user, realm)
        hash = self._records.get(key)
        if hash is None:
            return None
        if PY3:
            hash = hash.decode(self.encoding)
        return hash

    def set_hash(self, user, realm=None, hash=_UNSET):
        """
        semi-private helper which allows writing a hash directly;
        adds user & realm if needed.

        If ``self.default_realm`` has been set, this may be called
        with the syntax ``set_hash(user, hash)``,
        otherwise it must be called with all three arguments:
        ``set_hash(user, realm, hash)``.

        .. warning::
            does not (currently) do any validation of the hash string

        .. versionadded:: 1.7
        """
        if hash is _UNSET:
            realm, hash = (None, realm)
        if PY3 and isinstance(hash, str):
            hash = hash.encode(self.encoding)
        key = self._encode_key(user, realm)
        existing = self._set_record(key, hash)
        self._autosave()
        return existing

    @deprecated_method(deprecated='1.6', removed='1.8', replacement='get_hash')
    def find(self, user, realm):
        """return hash for user"""
        return self.get_hash(user, realm)

    def delete(self, user, realm=None):
        """Delete user's entry for specified realm.

        if realm is not specified, uses ``self.default_realm``.

        :returns:
            * ``True`` if user deleted,
            * ``False`` if user not found in realm.
        """
        key = self._encode_key(user, realm)
        try:
            del self._records[key]
        except KeyError:
            return False
        self._autosave()
        return True

    def delete_realm(self, realm):
        """Delete all users for specified realm.

        if realm is not specified, uses ``self.default_realm``.

        :returns: number of users deleted (0 if realm not found)
        """
        realm = self._encode_realm(realm)
        records = self._records
        keys = [key for key in records if key[1] == realm]
        for key in keys:
            del records[key]
        self._autosave()
        return len(keys)

    def check_password(self, user, realm=None, password=_UNSET):
        """Verify password for specified user + realm.

        If ``self.default_realm`` has been set, this may be called
        with the syntax ``check_password(user, password)``,
        otherwise it must be called with all three arguments:
        ``check_password(user, realm, password)``.

        :returns:
            * ``None`` if user or realm not found.
            * ``False`` if user found, but password does not match.
            * ``True`` if user found and password matches.

        .. versionchanged:: 1.6
            This method was previously called ``verify``, it was renamed
            to prevent ambiguity with the :class:`!CryptContext` method.
            The old alias is deprecated, and will be removed in Passlib 1.8.
        """
        if password is _UNSET:
            realm, password = (None, realm)
        user = self._encode_user(user)
        realm = self._encode_realm(realm)
        hash = self._records.get((user, realm))
        if hash is None:
            return None
        return htdigest.verify(password, hash, user, realm, encoding=self.encoding)

    @deprecated_method(deprecated='1.6', removed='1.8', replacement='check_password')
    def verify(self, user, realm, password):
        """verify password for user"""
        return self.check_password(user, realm, password)