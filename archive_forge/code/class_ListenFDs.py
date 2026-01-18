from os import getpid
from typing import Dict, List, Mapping, Optional, Sequence
from attrs import Factory, define
@define
class ListenFDs:
    """
    L{ListenFDs} provides access to file descriptors inherited from systemd.

    Typically L{ListenFDs.fromEnvironment} should be used to construct a new
    instance of L{ListenFDs}.

    @cvar _START: File descriptors inherited from systemd are always
        consecutively numbered, with a fixed lowest "starting" descriptor.  This
        gives the default starting descriptor.  Since this must agree with the
        value systemd is using, it typically should not be overridden.

    @ivar _descriptors: A C{list} of C{int} giving the descriptors which were
        inherited.

    @ivar _names: A L{Sequence} of C{str} giving the names of the descriptors
        which were inherited.
    """
    _descriptors: Sequence[int]
    _names: Sequence[str] = Factory(tuple)
    _START = 3

    @classmethod
    def fromEnvironment(cls, environ: Optional[Mapping[str, str]]=None, start: Optional[int]=None) -> 'ListenFDs':
        """
        @param environ: A dictionary-like object to inspect to discover
            inherited descriptors.  By default, L{None}, indicating that the
            real process environment should be inspected.  The default is
            suitable for typical usage.

        @param start: An integer giving the lowest value of an inherited
            descriptor systemd will give us.  By default, L{None}, indicating
            the known correct (that is, in agreement with systemd) value will be
            used.  The default is suitable for typical usage.

        @return: A new instance of C{cls} which can be used to look up the
            descriptors which have been inherited.
        """
        if environ is None:
            from os import environ as _environ
            environ = _environ
        if start is None:
            start = cls._START
        if str(getpid()) == environ.get('LISTEN_PID'):
            descriptors: List[int] = _parseDescriptors(start, environ)
            names: Sequence[str] = _parseNames(environ)
        else:
            descriptors = []
            names = ()
        if len(names) != len(descriptors):
            return cls([], ())
        return cls(descriptors, names)

    def inheritedDescriptors(self) -> List[int]:
        """
        @return: The configured descriptors.
        """
        return list(self._descriptors)

    def inheritedNamedDescriptors(self) -> Dict[str, int]:
        """
        @return: A mapping from the names of configured descriptors to
            their integer values.
        """
        return dict(zip(self._names, self._descriptors))