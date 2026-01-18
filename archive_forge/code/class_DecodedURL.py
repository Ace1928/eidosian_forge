import re
import sys
import string
import socket
from socket import AF_INET, AF_INET6
from typing import (
from unicodedata import normalize
from ._socket import inet_pton
from idna import encode as idna_encode, decode as idna_decode
class DecodedURL(object):
    """
    :class:`DecodedURL` is a type designed to act as a higher-level
    interface to :class:`URL` and the recommended type for most
    operations. By analogy, :class:`DecodedURL` is the
    :class:`unicode` to URL's :class:`bytes`.

    :class:`DecodedURL` automatically handles encoding and decoding
    all its components, such that all inputs and outputs are in a
    maximally-decoded state.  Note that this means, for some special
    cases, a URL may not "roundtrip" character-for-character, but this
    is considered a good tradeoff for the safety of automatic
    encoding.

    Otherwise, :class:`DecodedURL` has almost exactly the same API as
    :class:`URL`.

    Where applicable, a UTF-8 encoding is presumed. Be advised that
    some interactions can raise :exc:`UnicodeEncodeErrors` and
    :exc:`UnicodeDecodeErrors`, just like when working with
    bytestrings. Examples of such interactions include handling query
    strings encoding binary data, and paths containing segments with
    special characters encoded with codecs other than UTF-8.

    Args:
        url: A :class:`URL` object to wrap.
        lazy: Set to True to avoid pre-decode all parts of the URL to check for
            validity.
            Defaults to False.
        query_plus_is_space: + characters in the query string should be treated
            as spaces when decoding.  If unspecified, the default is taken from
            the scheme.

    .. note::

      The :class:`DecodedURL` initializer takes a :class:`URL` object,
      not URL components, like :class:`URL`. To programmatically
      construct a :class:`DecodedURL`, you can use this pattern:

        >>> print(DecodedURL().replace(scheme=u'https',
        ... host=u'pypi.org', path=(u'projects', u'hyperlink')).to_text())
        https://pypi.org/projects/hyperlink

    .. versionadded:: 18.0.0
    """

    def __init__(self, url=_EMPTY_URL, lazy=False, query_plus_is_space=None):
        self._url = url
        if query_plus_is_space is None:
            query_plus_is_space = url.scheme not in NO_QUERY_PLUS_SCHEMES
        self._query_plus_is_space = query_plus_is_space
        if not lazy:
            (self.host, self.userinfo, self.path, self.query, self.fragment)
        return

    @classmethod
    def from_text(cls, text, lazy=False, query_plus_is_space=None):
        """        Make a `DecodedURL` instance from any text string containing a URL.

        Args:
          text: Text containing the URL
          lazy: Whether to pre-decode all parts of the URL to check for
              validity.
              Defaults to True.
        """
        _url = URL.from_text(text)
        return cls(_url, lazy=lazy, query_plus_is_space=query_plus_is_space)

    @property
    def encoded_url(self):
        """Access the underlying :class:`URL` object, which has any special
        characters encoded.
        """
        return self._url

    def to_text(self, with_password=False):
        """Passthrough to :meth:`~hyperlink.URL.to_text()`"""
        return self._url.to_text(with_password)

    def to_uri(self):
        """Passthrough to :meth:`~hyperlink.URL.to_uri()`"""
        return self._url.to_uri()

    def to_iri(self):
        """Passthrough to :meth:`~hyperlink.URL.to_iri()`"""
        return self._url.to_iri()

    def _clone(self, url):
        return self.__class__(url, query_plus_is_space=self._query_plus_is_space)

    def click(self, href=u''):
        """Return a new DecodedURL wrapping the result of
        :meth:`~hyperlink.URL.click()`
        """
        if isinstance(href, DecodedURL):
            href = href._url
        return self._clone(self._url.click(href=href))

    def sibling(self, segment):
        """Automatically encode any reserved characters in *segment* and
        return a new `DecodedURL` wrapping the result of
        :meth:`~hyperlink.URL.sibling()`
        """
        return self._clone(self._url.sibling(_encode_reserved(segment)))

    def child(self, *segments):
        """Automatically encode any reserved characters in *segments* and
        return a new `DecodedURL` wrapping the result of
        :meth:`~hyperlink.URL.child()`.
        """
        if not segments:
            return self
        new_segs = [_encode_reserved(s) for s in segments]
        return self._clone(self._url.child(*new_segs))

    def normalize(self, scheme=True, host=True, path=True, query=True, fragment=True, userinfo=True, percents=True):
        """Return a new `DecodedURL` wrapping the result of
        :meth:`~hyperlink.URL.normalize()`
        """
        return self._clone(self._url.normalize(scheme, host, path, query, fragment, userinfo, percents))

    @property
    def absolute(self):
        return self._url.absolute

    @property
    def scheme(self):
        return self._url.scheme

    @property
    def host(self):
        return _decode_host(self._url.host)

    @property
    def port(self):
        return self._url.port

    @property
    def rooted(self):
        return self._url.rooted

    @property
    def path(self):
        if not hasattr(self, '_path'):
            self._path = tuple([_percent_decode(p, raise_subencoding_exc=True) for p in self._url.path])
        return self._path

    @property
    def query(self):
        if not hasattr(self, '_query'):
            if self._query_plus_is_space:
                predecode = _replace_plus
            else:
                predecode = _no_op
            self._query = cast(QueryPairs, tuple((tuple((_percent_decode(predecode(x), raise_subencoding_exc=True) if x is not None else None for x in (k, v))) for k, v in self._url.query)))
        return self._query

    @property
    def fragment(self):
        if not hasattr(self, '_fragment'):
            frag = self._url.fragment
            self._fragment = _percent_decode(frag, raise_subencoding_exc=True)
        return self._fragment

    @property
    def userinfo(self):
        if not hasattr(self, '_userinfo'):
            self._userinfo = cast(Union[Tuple[str], Tuple[str, str]], tuple(tuple((_percent_decode(p, raise_subencoding_exc=True) for p in self._url.userinfo.split(':', 1)))))
        return self._userinfo

    @property
    def user(self):
        return self.userinfo[0]

    @property
    def uses_netloc(self):
        return self._url.uses_netloc

    def replace(self, scheme=_UNSET, host=_UNSET, path=_UNSET, query=_UNSET, fragment=_UNSET, port=_UNSET, rooted=_UNSET, userinfo=_UNSET, uses_netloc=_UNSET):
        """While the signature is the same, this `replace()` differs a little
        from URL.replace. For instance, it accepts userinfo as a
        tuple, not as a string, handling the case of having a username
        containing a `:`. As with the rest of the methods on
        DecodedURL, if you pass a reserved character, it will be
        automatically encoded instead of an error being raised.
        """
        if path is not _UNSET:
            path = tuple((_encode_reserved(p) for p in path))
        if query is not _UNSET:
            query = cast(QueryPairs, tuple((tuple((_encode_reserved(x) if x is not None else None for x in (k, v))) for k, v in iter_pairs(query))))
        if userinfo is not _UNSET:
            if len(userinfo) > 2:
                raise ValueError('userinfo expected sequence of ["user"] or ["user", "password"], got %r' % (userinfo,))
            userinfo_text = u':'.join([_encode_reserved(p) for p in userinfo])
        else:
            userinfo_text = _UNSET
        new_url = self._url.replace(scheme=scheme, host=host, path=path, query=query, fragment=fragment, port=port, rooted=rooted, userinfo=userinfo_text, uses_netloc=uses_netloc)
        return self._clone(url=new_url)

    def get(self, name):
        """Get the value of all query parameters whose name matches *name*"""
        return [v for k, v in self.query if name == k]

    def add(self, name, value=None):
        """Return a new DecodedURL with the query parameter *name* and *value*
        added."""
        return self.replace(query=self.query + ((name, value),))

    def set(self, name, value=None):
        """Return a new DecodedURL with query parameter *name* set to *value*"""
        query = self.query
        q = [(k, v) for k, v in query if k != name]
        idx = next((i for i, (k, v) in enumerate(query) if k == name), -1)
        q[idx:idx] = [(name, value)]
        return self.replace(query=q)

    def remove(self, name, value=_UNSET, limit=None):
        """Return a new DecodedURL with query parameter *name* removed.

        Optionally also filter for *value*, as well as cap the number
        of parameters removed with *limit*.
        """
        if limit is None:
            if value is _UNSET:
                nq = [(k, v) for k, v in self.query if k != name]
            else:
                nq = [(k, v) for k, v in self.query if not (k == name and v == value)]
        else:
            nq, removed_count = ([], 0)
            for k, v in self.query:
                if k == name and (value is _UNSET or v == value) and (removed_count < limit):
                    removed_count += 1
                else:
                    nq.append((k, v))
        return self.replace(query=nq)

    def __repr__(self):
        cn = self.__class__.__name__
        return '%s(url=%r)' % (cn, self._url)

    def __str__(self):
        return str(self._url)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.normalize().to_uri() == other.normalize().to_uri()

    def __ne__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.__class__, self.scheme, self.userinfo, self.host, self.path, self.query, self.fragment, self.port, self.rooted, self.uses_netloc))
    asURI = to_uri
    asIRI = to_iri

    @classmethod
    def fromText(cls, s, lazy=False):
        return cls.from_text(s, lazy=lazy)

    def asText(self, includeSecrets=False):
        return self.to_text(with_password=includeSecrets)

    def __dir__(self):
        try:
            ret = object.__dir__(self)
        except AttributeError:
            ret = dir(self.__class__) + list(self.__dict__.keys())
        ret = sorted(set(ret) - set(['fromText', 'asURI', 'asIRI', 'asText']))
        return ret