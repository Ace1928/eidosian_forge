from __future__ import annotations
import dataclasses
import decimal
import json
import typing as t
import uuid
import weakref
from datetime import date
from werkzeug.http import http_date
class DefaultJSONProvider(JSONProvider):
    """Provide JSON operations using Python's built-in :mod:`json`
    library. Serializes the following additional data types:

    -   :class:`datetime.datetime` and :class:`datetime.date` are
        serialized to :rfc:`822` strings. This is the same as the HTTP
        date format.
    -   :class:`uuid.UUID` is serialized to a string.
    -   :class:`dataclasses.dataclass` is passed to
        :func:`dataclasses.asdict`.
    -   :class:`~markupsafe.Markup` (or any object with a ``__html__``
        method) will call the ``__html__`` method to get a string.
    """
    default: t.Callable[[t.Any], t.Any] = staticmethod(_default)
    'Apply this function to any object that :meth:`json.dumps` does\n    not know how to serialize. It should return a valid JSON type or\n    raise a ``TypeError``.\n    '
    ensure_ascii = True
    'Replace non-ASCII characters with escape sequences. This may be\n    more compatible with some clients, but can be disabled for better\n    performance and size.\n    '
    sort_keys = True
    'Sort the keys in any serialized dicts. This may be useful for\n    some caching situations, but can be disabled for better performance.\n    When enabled, keys must all be strings, they are not converted\n    before sorting.\n    '
    compact: bool | None = None
    'If ``True``, or ``None`` out of debug mode, the :meth:`response`\n    output will not add indentation, newlines, or spaces. If ``False``,\n    or ``None`` in debug mode, it will use a non-compact representation.\n    '
    mimetype = 'application/json'
    'The mimetype set in :meth:`response`.'

    def dumps(self, obj: t.Any, **kwargs: t.Any) -> str:
        """Serialize data as JSON to a string.

        Keyword arguments are passed to :func:`json.dumps`. Sets some
        parameter defaults from the :attr:`default`,
        :attr:`ensure_ascii`, and :attr:`sort_keys` attributes.

        :param obj: The data to serialize.
        :param kwargs: Passed to :func:`json.dumps`.
        """
        kwargs.setdefault('default', self.default)
        kwargs.setdefault('ensure_ascii', self.ensure_ascii)
        kwargs.setdefault('sort_keys', self.sort_keys)
        return json.dumps(obj, **kwargs)

    def loads(self, s: str | bytes, **kwargs: t.Any) -> t.Any:
        """Deserialize data as JSON from a string or bytes.

        :param s: Text or UTF-8 bytes.
        :param kwargs: Passed to :func:`json.loads`.
        """
        return json.loads(s, **kwargs)

    def response(self, *args: t.Any, **kwargs: t.Any) -> Response:
        """Serialize the given arguments as JSON, and return a
        :class:`~flask.Response` object with it. The response mimetype
        will be "application/json" and can be changed with
        :attr:`mimetype`.

        If :attr:`compact` is ``False`` or debug mode is enabled, the
        output will be formatted to be easier to read.

        Either positional or keyword arguments can be given, not both.
        If no arguments are given, ``None`` is serialized.

        :param args: A single value to serialize, or multiple values to
            treat as a list to serialize.
        :param kwargs: Treat as a dict to serialize.
        """
        obj = self._prepare_response_obj(args, kwargs)
        dump_args: dict[str, t.Any] = {}
        if self.compact is None and self._app.debug or self.compact is False:
            dump_args.setdefault('indent', 2)
        else:
            dump_args.setdefault('separators', (',', ':'))
        return self._app.response_class(f'{self.dumps(obj, **dump_args)}\n', mimetype=self.mimetype)