from __future__ import annotations
from collections.abc import MutableSet
from copy import deepcopy
from .. import exceptions
from .._internal import _missing
from .mixins import ImmutableDictMixin
from .mixins import ImmutableListMixin
from .mixins import ImmutableMultiDictMixin
from .mixins import UpdateDictMixin
from .. import http
class HeaderSet(MutableSet):
    """Similar to the :class:`ETags` class this implements a set-like structure.
    Unlike :class:`ETags` this is case insensitive and used for vary, allow, and
    content-language headers.

    If not constructed using the :func:`parse_set_header` function the
    instantiation works like this:

    >>> hs = HeaderSet(['foo', 'bar', 'baz'])
    >>> hs
    HeaderSet(['foo', 'bar', 'baz'])
    """

    def __init__(self, headers=None, on_update=None):
        self._headers = list(headers or ())
        self._set = {x.lower() for x in self._headers}
        self.on_update = on_update

    def add(self, header):
        """Add a new header to the set."""
        self.update((header,))

    def remove(self, header):
        """Remove a header from the set.  This raises an :exc:`KeyError` if the
        header is not in the set.

        .. versionchanged:: 0.5
            In older versions a :exc:`IndexError` was raised instead of a
            :exc:`KeyError` if the object was missing.

        :param header: the header to be removed.
        """
        key = header.lower()
        if key not in self._set:
            raise KeyError(header)
        self._set.remove(key)
        for idx, key in enumerate(self._headers):
            if key.lower() == header:
                del self._headers[idx]
                break
        if self.on_update is not None:
            self.on_update(self)

    def update(self, iterable):
        """Add all the headers from the iterable to the set.

        :param iterable: updates the set with the items from the iterable.
        """
        inserted_any = False
        for header in iterable:
            key = header.lower()
            if key not in self._set:
                self._headers.append(header)
                self._set.add(key)
                inserted_any = True
        if inserted_any and self.on_update is not None:
            self.on_update(self)

    def discard(self, header):
        """Like :meth:`remove` but ignores errors.

        :param header: the header to be discarded.
        """
        try:
            self.remove(header)
        except KeyError:
            pass

    def find(self, header):
        """Return the index of the header in the set or return -1 if not found.

        :param header: the header to be looked up.
        """
        header = header.lower()
        for idx, item in enumerate(self._headers):
            if item.lower() == header:
                return idx
        return -1

    def index(self, header):
        """Return the index of the header in the set or raise an
        :exc:`IndexError`.

        :param header: the header to be looked up.
        """
        rv = self.find(header)
        if rv < 0:
            raise IndexError(header)
        return rv

    def clear(self):
        """Clear the set."""
        self._set.clear()
        del self._headers[:]
        if self.on_update is not None:
            self.on_update(self)

    def as_set(self, preserve_casing=False):
        """Return the set as real python set type.  When calling this, all
        the items are converted to lowercase and the ordering is lost.

        :param preserve_casing: if set to `True` the items in the set returned
                                will have the original case like in the
                                :class:`HeaderSet`, otherwise they will
                                be lowercase.
        """
        if preserve_casing:
            return set(self._headers)
        return set(self._set)

    def to_header(self):
        """Convert the header set into an HTTP header string."""
        return ', '.join(map(http.quote_header_value, self._headers))

    def __getitem__(self, idx):
        return self._headers[idx]

    def __delitem__(self, idx):
        rv = self._headers.pop(idx)
        self._set.remove(rv.lower())
        if self.on_update is not None:
            self.on_update(self)

    def __setitem__(self, idx, value):
        old = self._headers[idx]
        self._set.remove(old.lower())
        self._headers[idx] = value
        self._set.add(value.lower())
        if self.on_update is not None:
            self.on_update(self)

    def __contains__(self, header):
        return header.lower() in self._set

    def __len__(self):
        return len(self._set)

    def __iter__(self):
        return iter(self._headers)

    def __bool__(self):
        return bool(self._set)

    def __str__(self):
        return self.to_header()

    def __repr__(self):
        return f'{type(self).__name__}({self._headers!r})'