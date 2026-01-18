from __future__ import annotations
import dataclasses
import mimetypes
import sys
import typing as t
from collections import defaultdict
from datetime import datetime
from io import BytesIO
from itertools import chain
from random import random
from tempfile import TemporaryFile
from time import time
from urllib.parse import unquote
from urllib.parse import urlsplit
from urllib.parse import urlunsplit
from ._internal import _get_environ
from ._internal import _wsgi_decoding_dance
from ._internal import _wsgi_encoding_dance
from .datastructures import Authorization
from .datastructures import CallbackDict
from .datastructures import CombinedMultiDict
from .datastructures import EnvironHeaders
from .datastructures import FileMultiDict
from .datastructures import Headers
from .datastructures import MultiDict
from .http import dump_cookie
from .http import dump_options_header
from .http import parse_cookie
from .http import parse_date
from .http import parse_options_header
from .sansio.multipart import Data
from .sansio.multipart import Epilogue
from .sansio.multipart import Field
from .sansio.multipart import File
from .sansio.multipart import MultipartEncoder
from .sansio.multipart import Preamble
from .urls import _urlencode
from .urls import iri_to_uri
from .utils import cached_property
from .utils import get_content_type
from .wrappers.request import Request
from .wrappers.response import Response
from .wsgi import ClosingIterator
from .wsgi import get_current_url
class EnvironBuilder:
    """This class can be used to conveniently create a WSGI environment
    for testing purposes.  It can be used to quickly create WSGI environments
    or request objects from arbitrary data.

    The signature of this class is also used in some other places as of
    Werkzeug 0.5 (:func:`create_environ`, :meth:`Response.from_values`,
    :meth:`Client.open`).  Because of this most of the functionality is
    available through the constructor alone.

    Files and regular form data can be manipulated independently of each
    other with the :attr:`form` and :attr:`files` attributes, but are
    passed with the same argument to the constructor: `data`.

    `data` can be any of these values:

    -   a `str` or `bytes` object: The object is converted into an
        :attr:`input_stream`, the :attr:`content_length` is set and you have to
        provide a :attr:`content_type`.
    -   a `dict` or :class:`MultiDict`: The keys have to be strings. The values
        have to be either any of the following objects, or a list of any of the
        following objects:

        -   a :class:`file`-like object:  These are converted into
            :class:`FileStorage` objects automatically.
        -   a `tuple`:  The :meth:`~FileMultiDict.add_file` method is called
            with the key and the unpacked `tuple` items as positional
            arguments.
        -   a `str`:  The string is set as form data for the associated key.
    -   a file-like object: The object content is loaded in memory and then
        handled like a regular `str` or a `bytes`.

    :param path: the path of the request.  In the WSGI environment this will
                 end up as `PATH_INFO`.  If the `query_string` is not defined
                 and there is a question mark in the `path` everything after
                 it is used as query string.
    :param base_url: the base URL is a URL that is used to extract the WSGI
                     URL scheme, host (server name + server port) and the
                     script root (`SCRIPT_NAME`).
    :param query_string: an optional string or dict with URL parameters.
    :param method: the HTTP method to use, defaults to `GET`.
    :param input_stream: an optional input stream.  Do not specify this and
                         `data`.  As soon as an input stream is set you can't
                         modify :attr:`args` and :attr:`files` unless you
                         set the :attr:`input_stream` to `None` again.
    :param content_type: The content type for the request.  As of 0.5 you
                         don't have to provide this when specifying files
                         and form data via `data`.
    :param content_length: The content length for the request.  You don't
                           have to specify this when providing data via
                           `data`.
    :param errors_stream: an optional error stream that is used for
                          `wsgi.errors`.  Defaults to :data:`stderr`.
    :param multithread: controls `wsgi.multithread`.  Defaults to `False`.
    :param multiprocess: controls `wsgi.multiprocess`.  Defaults to `False`.
    :param run_once: controls `wsgi.run_once`.  Defaults to `False`.
    :param headers: an optional list or :class:`Headers` object of headers.
    :param data: a string or dict of form data or a file-object.
                 See explanation above.
    :param json: An object to be serialized and assigned to ``data``.
        Defaults the content type to ``"application/json"``.
        Serialized with the function assigned to :attr:`json_dumps`.
    :param environ_base: an optional dict of environment defaults.
    :param environ_overrides: an optional dict of environment overrides.
    :param auth: An authorization object to use for the
        ``Authorization`` header value. A ``(username, password)`` tuple
        is a shortcut for ``Basic`` authorization.

    .. versionchanged:: 3.0
        The ``charset`` parameter was removed.

    .. versionchanged:: 2.1
        ``CONTENT_TYPE`` and ``CONTENT_LENGTH`` are not duplicated as
        header keys in the environ.

    .. versionchanged:: 2.0
        ``REQUEST_URI`` and ``RAW_URI`` is the full raw URI including
        the query string, not only the path.

    .. versionchanged:: 2.0
        The default :attr:`request_class` is ``Request`` instead of
        ``BaseRequest``.

    .. versionadded:: 2.0
       Added the ``auth`` parameter.

    .. versionadded:: 0.15
        The ``json`` param and :meth:`json_dumps` method.

    .. versionadded:: 0.15
        The environ has keys ``REQUEST_URI`` and ``RAW_URI`` containing
        the path before percent-decoding. This is not part of the WSGI
        PEP, but many WSGI servers include it.

    .. versionchanged:: 0.6
       ``path`` and ``base_url`` can now be unicode strings that are
       encoded with :func:`iri_to_uri`.
    """
    server_protocol = 'HTTP/1.1'
    wsgi_version = (1, 0)
    request_class = Request
    import json
    json_dumps = staticmethod(json.dumps)
    del json
    _args: MultiDict | None
    _query_string: str | None
    _input_stream: t.IO[bytes] | None
    _form: MultiDict | None
    _files: FileMultiDict | None

    def __init__(self, path: str='/', base_url: str | None=None, query_string: t.Mapping[str, str] | str | None=None, method: str='GET', input_stream: t.IO[bytes] | None=None, content_type: str | None=None, content_length: int | None=None, errors_stream: t.IO[str] | None=None, multithread: bool=False, multiprocess: bool=False, run_once: bool=False, headers: Headers | t.Iterable[tuple[str, str]] | None=None, data: None | (t.IO[bytes] | str | bytes | t.Mapping[str, t.Any])=None, environ_base: t.Mapping[str, t.Any] | None=None, environ_overrides: t.Mapping[str, t.Any] | None=None, mimetype: str | None=None, json: t.Mapping[str, t.Any] | None=None, auth: Authorization | tuple[str, str] | None=None) -> None:
        if query_string is not None and '?' in path:
            raise ValueError('Query string is defined in the path and as an argument')
        request_uri = urlsplit(path)
        if query_string is None and '?' in path:
            query_string = request_uri.query
        self.path = iri_to_uri(request_uri.path)
        self.request_uri = path
        if base_url is not None:
            base_url = iri_to_uri(base_url)
        self.base_url = base_url
        if isinstance(query_string, str):
            self.query_string = query_string
        else:
            if query_string is None:
                query_string = MultiDict()
            elif not isinstance(query_string, MultiDict):
                query_string = MultiDict(query_string)
            self.args = query_string
        self.method = method
        if headers is None:
            headers = Headers()
        elif not isinstance(headers, Headers):
            headers = Headers(headers)
        self.headers = headers
        if content_type is not None:
            self.content_type = content_type
        if errors_stream is None:
            errors_stream = sys.stderr
        self.errors_stream = errors_stream
        self.multithread = multithread
        self.multiprocess = multiprocess
        self.run_once = run_once
        self.environ_base = environ_base
        self.environ_overrides = environ_overrides
        self.input_stream = input_stream
        self.content_length = content_length
        self.closed = False
        if auth is not None:
            if isinstance(auth, tuple):
                auth = Authorization('basic', {'username': auth[0], 'password': auth[1]})
            self.headers.set('Authorization', auth.to_header())
        if json is not None:
            if data is not None:
                raise TypeError("can't provide both json and data")
            data = self.json_dumps(json)
            if self.content_type is None:
                self.content_type = 'application/json'
        if data:
            if input_stream is not None:
                raise TypeError("can't provide input stream and data")
            if hasattr(data, 'read'):
                data = data.read()
            if isinstance(data, str):
                data = data.encode()
            if isinstance(data, bytes):
                self.input_stream = BytesIO(data)
                if self.content_length is None:
                    self.content_length = len(data)
            else:
                for key, value in _iter_data(data):
                    if isinstance(value, (tuple, dict)) or hasattr(value, 'read'):
                        self._add_file_from_data(key, value)
                    else:
                        self.form.setlistdefault(key).append(value)
        if mimetype is not None:
            self.mimetype = mimetype

    @classmethod
    def from_environ(cls, environ: WSGIEnvironment, **kwargs: t.Any) -> EnvironBuilder:
        """Turn an environ dict back into a builder. Any extra kwargs
        override the args extracted from the environ.

        .. versionchanged:: 2.0
            Path and query values are passed through the WSGI decoding
            dance to avoid double encoding.

        .. versionadded:: 0.15
        """
        headers = Headers(EnvironHeaders(environ))
        out = {'path': _wsgi_decoding_dance(environ['PATH_INFO']), 'base_url': cls._make_base_url(environ['wsgi.url_scheme'], headers.pop('Host'), _wsgi_decoding_dance(environ['SCRIPT_NAME'])), 'query_string': _wsgi_decoding_dance(environ['QUERY_STRING']), 'method': environ['REQUEST_METHOD'], 'input_stream': environ['wsgi.input'], 'content_type': headers.pop('Content-Type', None), 'content_length': headers.pop('Content-Length', None), 'errors_stream': environ['wsgi.errors'], 'multithread': environ['wsgi.multithread'], 'multiprocess': environ['wsgi.multiprocess'], 'run_once': environ['wsgi.run_once'], 'headers': headers}
        out.update(kwargs)
        return cls(**out)

    def _add_file_from_data(self, key: str, value: t.IO[bytes] | tuple[t.IO[bytes], str] | tuple[t.IO[bytes], str, str]) -> None:
        """Called in the EnvironBuilder to add files from the data dict."""
        if isinstance(value, tuple):
            self.files.add_file(key, *value)
        else:
            self.files.add_file(key, value)

    @staticmethod
    def _make_base_url(scheme: str, host: str, script_root: str) -> str:
        return urlunsplit((scheme, host, script_root, '', '')).rstrip('/') + '/'

    @property
    def base_url(self) -> str:
        """The base URL is used to extract the URL scheme, host name,
        port, and root path.
        """
        return self._make_base_url(self.url_scheme, self.host, self.script_root)

    @base_url.setter
    def base_url(self, value: str | None) -> None:
        if value is None:
            scheme = 'http'
            netloc = 'localhost'
            script_root = ''
        else:
            scheme, netloc, script_root, qs, anchor = urlsplit(value)
            if qs or anchor:
                raise ValueError('base url must not contain a query string or fragment')
        self.script_root = script_root.rstrip('/')
        self.host = netloc
        self.url_scheme = scheme

    @property
    def content_type(self) -> str | None:
        """The content type for the request.  Reflected from and to
        the :attr:`headers`.  Do not set if you set :attr:`files` or
        :attr:`form` for auto detection.
        """
        ct = self.headers.get('Content-Type')
        if ct is None and (not self._input_stream):
            if self._files:
                return 'multipart/form-data'
            if self._form:
                return 'application/x-www-form-urlencoded'
            return None
        return ct

    @content_type.setter
    def content_type(self, value: str | None) -> None:
        if value is None:
            self.headers.pop('Content-Type', None)
        else:
            self.headers['Content-Type'] = value

    @property
    def mimetype(self) -> str | None:
        """The mimetype (content type without charset etc.)

        .. versionadded:: 0.14
        """
        ct = self.content_type
        return ct.split(';')[0].strip() if ct else None

    @mimetype.setter
    def mimetype(self, value: str) -> None:
        self.content_type = get_content_type(value, 'utf-8')

    @property
    def mimetype_params(self) -> t.Mapping[str, str]:
        """The mimetype parameters as dict.  For example if the
        content type is ``text/html; charset=utf-8`` the params would be
        ``{'charset': 'utf-8'}``.

        .. versionadded:: 0.14
        """

        def on_update(d: CallbackDict) -> None:
            self.headers['Content-Type'] = dump_options_header(self.mimetype, d)
        d = parse_options_header(self.headers.get('content-type', ''))[1]
        return CallbackDict(d, on_update)

    @property
    def content_length(self) -> int | None:
        """The content length as integer.  Reflected from and to the
        :attr:`headers`.  Do not set if you set :attr:`files` or
        :attr:`form` for auto detection.
        """
        return self.headers.get('Content-Length', type=int)

    @content_length.setter
    def content_length(self, value: int | None) -> None:
        if value is None:
            self.headers.pop('Content-Length', None)
        else:
            self.headers['Content-Length'] = str(value)

    def _get_form(self, name: str, storage: type[_TAnyMultiDict]) -> _TAnyMultiDict:
        """Common behavior for getting the :attr:`form` and
        :attr:`files` properties.

        :param name: Name of the internal cached attribute.
        :param storage: Storage class used for the data.
        """
        if self.input_stream is not None:
            raise AttributeError('an input stream is defined')
        rv = getattr(self, name)
        if rv is None:
            rv = storage()
            setattr(self, name, rv)
        return rv

    def _set_form(self, name: str, value: MultiDict) -> None:
        """Common behavior for setting the :attr:`form` and
        :attr:`files` properties.

        :param name: Name of the internal cached attribute.
        :param value: Value to assign to the attribute.
        """
        self._input_stream = None
        setattr(self, name, value)

    @property
    def form(self) -> MultiDict:
        """A :class:`MultiDict` of form values."""
        return self._get_form('_form', MultiDict)

    @form.setter
    def form(self, value: MultiDict) -> None:
        self._set_form('_form', value)

    @property
    def files(self) -> FileMultiDict:
        """A :class:`FileMultiDict` of uploaded files. Use
        :meth:`~FileMultiDict.add_file` to add new files.
        """
        return self._get_form('_files', FileMultiDict)

    @files.setter
    def files(self, value: FileMultiDict) -> None:
        self._set_form('_files', value)

    @property
    def input_stream(self) -> t.IO[bytes] | None:
        """An optional input stream. This is mutually exclusive with
        setting :attr:`form` and :attr:`files`, setting it will clear
        those. Do not provide this if the method is not ``POST`` or
        another method that has a body.
        """
        return self._input_stream

    @input_stream.setter
    def input_stream(self, value: t.IO[bytes] | None) -> None:
        self._input_stream = value
        self._form = None
        self._files = None

    @property
    def query_string(self) -> str:
        """The query string.  If you set this to a string
        :attr:`args` will no longer be available.
        """
        if self._query_string is None:
            if self._args is not None:
                return _urlencode(self._args)
            return ''
        return self._query_string

    @query_string.setter
    def query_string(self, value: str | None) -> None:
        self._query_string = value
        self._args = None

    @property
    def args(self) -> MultiDict:
        """The URL arguments as :class:`MultiDict`."""
        if self._query_string is not None:
            raise AttributeError('a query string is defined')
        if self._args is None:
            self._args = MultiDict()
        return self._args

    @args.setter
    def args(self, value: MultiDict | None) -> None:
        self._query_string = None
        self._args = value

    @property
    def server_name(self) -> str:
        """The server name (read-only, use :attr:`host` to set)"""
        return self.host.split(':', 1)[0]

    @property
    def server_port(self) -> int:
        """The server port as integer (read-only, use :attr:`host` to set)"""
        pieces = self.host.split(':', 1)
        if len(pieces) == 2:
            try:
                return int(pieces[1])
            except ValueError:
                pass
        if self.url_scheme == 'https':
            return 443
        return 80

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def close(self) -> None:
        """Closes all files.  If you put real :class:`file` objects into the
        :attr:`files` dict you can call this method to automatically close
        them all in one go.
        """
        if self.closed:
            return
        try:
            files = self.files.values()
        except AttributeError:
            files = ()
        for f in files:
            try:
                f.close()
            except Exception:
                pass
        self.closed = True

    def get_environ(self) -> WSGIEnvironment:
        """Return the built environ.

        .. versionchanged:: 0.15
            The content type and length headers are set based on
            input stream detection. Previously this only set the WSGI
            keys.
        """
        input_stream = self.input_stream
        content_length = self.content_length
        mimetype = self.mimetype
        content_type = self.content_type
        if input_stream is not None:
            start_pos = input_stream.tell()
            input_stream.seek(0, 2)
            end_pos = input_stream.tell()
            input_stream.seek(start_pos)
            content_length = end_pos - start_pos
        elif mimetype == 'multipart/form-data':
            input_stream, content_length, boundary = stream_encode_multipart(CombinedMultiDict([self.form, self.files]))
            content_type = f'{mimetype}; boundary="{boundary}"'
        elif mimetype == 'application/x-www-form-urlencoded':
            form_encoded = _urlencode(self.form).encode('ascii')
            content_length = len(form_encoded)
            input_stream = BytesIO(form_encoded)
        else:
            input_stream = BytesIO()
        result: WSGIEnvironment = {}
        if self.environ_base:
            result.update(self.environ_base)

        def _path_encode(x: str) -> str:
            return _wsgi_encoding_dance(unquote(x))
        raw_uri = _wsgi_encoding_dance(self.request_uri)
        result.update({'REQUEST_METHOD': self.method, 'SCRIPT_NAME': _path_encode(self.script_root), 'PATH_INFO': _path_encode(self.path), 'QUERY_STRING': _wsgi_encoding_dance(self.query_string), 'REQUEST_URI': raw_uri, 'RAW_URI': raw_uri, 'SERVER_NAME': self.server_name, 'SERVER_PORT': str(self.server_port), 'HTTP_HOST': self.host, 'SERVER_PROTOCOL': self.server_protocol, 'wsgi.version': self.wsgi_version, 'wsgi.url_scheme': self.url_scheme, 'wsgi.input': input_stream, 'wsgi.errors': self.errors_stream, 'wsgi.multithread': self.multithread, 'wsgi.multiprocess': self.multiprocess, 'wsgi.run_once': self.run_once})
        headers = self.headers.copy()
        headers.remove('Content-Type')
        headers.remove('Content-Length')
        if content_type is not None:
            result['CONTENT_TYPE'] = content_type
        if content_length is not None:
            result['CONTENT_LENGTH'] = str(content_length)
        combined_headers = defaultdict(list)
        for key, value in headers.to_wsgi_list():
            combined_headers[f'HTTP_{key.upper().replace('-', '_')}'].append(value)
        for key, values in combined_headers.items():
            result[key] = ', '.join(values)
        if self.environ_overrides:
            result.update(self.environ_overrides)
        return result

    def get_request(self, cls: type[Request] | None=None) -> Request:
        """Returns a request with the data.  If the request class is not
        specified :attr:`request_class` is used.

        :param cls: The request wrapper to use.
        """
        if cls is None:
            cls = self.request_class
        return cls(self.get_environ())