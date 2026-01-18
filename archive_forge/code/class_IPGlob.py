from netaddr.core import AddrFormatError, AddrConversionError
from netaddr.ip import IPRange, IPAddress, IPNetwork, iprange_to_cidrs
class IPGlob(IPRange):
    """
    Represents an IP address range using a glob-style syntax ``x.x.x-y.*``

    Individual octets can be represented using the following shortcuts :

        1. ``*`` - the asterisk octet (represents values ``0`` through ``255``)
        2. ``x-y`` - the hyphenated octet (represents values ``x`` through ``y``)

    A few basic rules also apply :

        1. ``x`` must always be less than ``y``, therefore :

        - ``x`` can only be ``0`` through ``254``
        - ``y`` can only be ``1`` through ``255``

        2. only one hyphenated octet per IP glob is allowed
        3. only asterisks are permitted after a hyphenated octet

    Examples:

    +------------------+------------------------------+
    | IP glob          | Description                  |
    +==================+==============================+
    | ``192.0.2.1``    | a single address             |
    +------------------+------------------------------+
    | ``192.0.2.0-31`` | 32 addresses                 |
    +------------------+------------------------------+
    | ``192.0.2.*``    | 256 addresses                |
    +------------------+------------------------------+
    | ``192.0.2-3.*``  | 512 addresses                |
    +------------------+------------------------------+
    | ``192.0-1.*.*``  | 131,072 addresses            |
    +------------------+------------------------------+
    | ``*.*.*.*``      | the whole IPv4 address space |
    +------------------+------------------------------+

    .. note ::     IP glob ranges are not directly equivalent to CIDR blocks.     They can represent address ranges that do not fall on strict bit mask     boundaries. They are suitable for use in configuration files, being     more obvious and readable than their CIDR counterparts, especially for     admins and end users with little or no networking knowledge or     experience. All CIDR addresses can always be represented as IP globs     but the reverse is not always true.
    """
    __slots__ = ('_glob',)

    def __init__(self, ipglob):
        start, end = glob_to_iptuple(ipglob)
        super(IPGlob, self).__init__(start, end)
        self.glob = iprange_to_globs(self._start, self._end)[0]

    def __getstate__(self):
        """:return: Pickled state of an `IPGlob` object."""
        return super(IPGlob, self).__getstate__()

    def __setstate__(self, state):
        """:param state: data used to unpickle a pickled `IPGlob` object."""
        super(IPGlob, self).__setstate__(state)
        self.glob = iprange_to_globs(self._start, self._end)[0]

    def _get_glob(self):
        return self._glob

    def _set_glob(self, ipglob):
        self._start, self._end = glob_to_iptuple(ipglob)
        self._glob = iprange_to_globs(self._start, self._end)[0]
    glob = property(_get_glob, _set_glob, None, 'an arbitrary IP address range in glob format.')

    def __str__(self):
        """:return: IP glob in common representational format."""
        return '%s' % self.glob

    def __repr__(self):
        """:return: Python statement to create an equivalent object"""
        return "%s('%s')" % (self.__class__.__name__, self.glob)