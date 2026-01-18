import contextlib
import io
import os
from uuid import uuid4
import requests
from .._compat import fields
class MultipartEncoder(object):
    """

    The ``MultipartEncoder`` object is a generic interface to the engine that
    will create a ``multipart/form-data`` body for you.

    The basic usage is:

    .. code-block:: python

        import requests
        from requests_toolbelt import MultipartEncoder

        encoder = MultipartEncoder({'field': 'value',
                                    'other_field': 'other_value'})
        r = requests.post('https://httpbin.org/post', data=encoder,
                          headers={'Content-Type': encoder.content_type})

    If you do not need to take advantage of streaming the post body, you can
    also do:

    .. code-block:: python

        r = requests.post('https://httpbin.org/post',
                          data=encoder.to_string(),
                          headers={'Content-Type': encoder.content_type})

    If you want the encoder to use a specific order, you can use an
    OrderedDict or more simply, a list of tuples:

    .. code-block:: python

        encoder = MultipartEncoder([('field', 'value'),
                                    ('other_field', 'other_value')])

    .. versionchanged:: 0.4.0

    You can also provide tuples as part values as you would provide them to
    requests' ``files`` parameter.

    .. code-block:: python

        encoder = MultipartEncoder({
            'field': ('file_name', b'{"a": "b"}', 'application/json',
                      {'X-My-Header': 'my-value'})
        ])

    .. warning::

        This object will end up directly in :mod:`httplib`. Currently,
        :mod:`httplib` has a hard-coded read size of **8192 bytes**. This
        means that it will loop until the file has been read and your upload
        could take a while. This is **not** a bug in requests. A feature is
        being considered for this object to allow you, the user, to specify
        what size should be returned on a read. If you have opinions on this,
        please weigh in on `this issue`_.

    .. _this issue:
        https://github.com/requests/toolbelt/issues/75

    """

    def __init__(self, fields, boundary=None, encoding='utf-8'):
        self.boundary_value = boundary or uuid4().hex
        self.boundary = '--{}'.format(self.boundary_value)
        self.encoding = encoding
        self._encoded_boundary = b''.join([encode_with(self.boundary, self.encoding), encode_with('\r\n', self.encoding)])
        self.fields = fields
        self.finished = False
        self.parts = []
        self._iter_parts = iter([])
        self._current_part = None
        self._len = None
        self._buffer = CustomBytesIO(encoding=encoding)
        self._prepare_parts()
        self._write_boundary()

    @property
    def len(self):
        """Length of the multipart/form-data body.

        requests will first attempt to get the length of the body by calling
        ``len(body)`` and then by checking for the ``len`` attribute.

        On 32-bit systems, the ``__len__`` method cannot return anything
        larger than an integer (in C) can hold. If the total size of the body
        is even slightly larger than 4GB users will see an OverflowError. This
        manifested itself in `bug #80`_.

        As such, we now calculate the length lazily as a property.

        .. _bug #80:
            https://github.com/requests/toolbelt/issues/80
        """
        return self._len or self._calculate_length()

    def __repr__(self):
        return '<MultipartEncoder: {!r}>'.format(self.fields)

    def _calculate_length(self):
        """
        This uses the parts to calculate the length of the body.

        This returns the calculated length so __len__ can be lazy.
        """
        boundary_len = len(self.boundary)
        self._len = sum((boundary_len + total_len(p) + 4 for p in self.parts)) + boundary_len + 4
        return self._len

    def _calculate_load_amount(self, read_size):
        """This calculates how many bytes need to be added to the buffer.

        When a consumer read's ``x`` from the buffer, there are two cases to
        satisfy:

            1. Enough data in the buffer to return the requested amount
            2. Not enough data

        This function uses the amount of unread bytes in the buffer and
        determines how much the Encoder has to load before it can return the
        requested amount of bytes.

        :param int read_size: the number of bytes the consumer requests
        :returns: int -- the number of bytes that must be loaded into the
            buffer before the read can be satisfied. This will be strictly
            non-negative
        """
        amount = read_size - total_len(self._buffer)
        return amount if amount > 0 else 0

    def _load(self, amount):
        """Load ``amount`` number of bytes into the buffer."""
        self._buffer.smart_truncate()
        part = self._current_part or self._next_part()
        while amount == -1 or amount > 0:
            written = 0
            if part and (not part.bytes_left_to_write()):
                written += self._write(b'\r\n')
                written += self._write_boundary()
                part = self._next_part()
            if not part:
                written += self._write_closing_boundary()
                self.finished = True
                break
            written += part.write_to(self._buffer, amount)
            if amount != -1:
                amount -= written

    def _next_part(self):
        try:
            p = self._current_part = next(self._iter_parts)
        except StopIteration:
            p = None
        return p

    def _iter_fields(self):
        _fields = self.fields
        if hasattr(self.fields, 'items'):
            _fields = list(self.fields.items())
        for k, v in _fields:
            file_name = None
            file_type = None
            file_headers = None
            if isinstance(v, (list, tuple)):
                if len(v) == 2:
                    file_name, file_pointer = v
                elif len(v) == 3:
                    file_name, file_pointer, file_type = v
                else:
                    file_name, file_pointer, file_type, file_headers = v
            else:
                file_pointer = v
            field = fields.RequestField(name=k, data=file_pointer, filename=file_name, headers=file_headers)
            field.make_multipart(content_type=file_type)
            yield field

    def _prepare_parts(self):
        """This uses the fields provided by the user and creates Part objects.

        It populates the `parts` attribute and uses that to create a
        generator for iteration.
        """
        enc = self.encoding
        self.parts = [Part.from_field(f, enc) for f in self._iter_fields()]
        self._iter_parts = iter(self.parts)

    def _write(self, bytes_to_write):
        """Write the bytes to the end of the buffer.

        :param bytes bytes_to_write: byte-string (or bytearray) to append to
            the buffer
        :returns: int -- the number of bytes written
        """
        return self._buffer.append(bytes_to_write)

    def _write_boundary(self):
        """Write the boundary to the end of the buffer."""
        return self._write(self._encoded_boundary)

    def _write_closing_boundary(self):
        """Write the bytes necessary to finish a multipart/form-data body."""
        with reset(self._buffer):
            self._buffer.seek(-2, 2)
            self._buffer.write(b'--\r\n')
        return 2

    def _write_headers(self, headers):
        """Write the current part's headers to the buffer."""
        return self._write(encode_with(headers, self.encoding))

    @property
    def content_type(self):
        return str('multipart/form-data; boundary={}'.format(self.boundary_value))

    def to_string(self):
        """Return the entirety of the data in the encoder.

        .. note::

            This simply reads all of the data it can. If you have started
            streaming or reading data from the encoder, this method will only
            return whatever data is left in the encoder.

        .. note::

            This method affects the internal state of the encoder. Calling
            this method will exhaust the encoder.

        :returns: the multipart message
        :rtype: bytes
        """
        return self.read()

    def read(self, size=-1):
        """Read data from the streaming encoder.

        :param int size: (optional), If provided, ``read`` will return exactly
            that many bytes. If it is not provided, it will return the
            remaining bytes.
        :returns: bytes
        """
        if self.finished:
            return self._buffer.read(size)
        bytes_to_load = size
        if bytes_to_load != -1 and bytes_to_load is not None:
            bytes_to_load = self._calculate_load_amount(int(size))
        self._load(bytes_to_load)
        return self._buffer.read(size)