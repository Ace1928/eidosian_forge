import contextlib
import io
import os
from uuid import uuid4
import requests
from .._compat import fields
class MultipartEncoderMonitor(object):
    """
    An object used to monitor the progress of a :class:`MultipartEncoder`.

    The :class:`MultipartEncoder` should only be responsible for preparing and
    streaming the data. For anyone who wishes to monitor it, they shouldn't be
    using that instance to manage that as well. Using this class, they can
    monitor an encoder and register a callback. The callback receives the
    instance of the monitor.

    To use this monitor, you construct your :class:`MultipartEncoder` as you
    normally would.

    .. code-block:: python

        from requests_toolbelt import (MultipartEncoder,
                                       MultipartEncoderMonitor)
        import requests

        def callback(monitor):
            # Do something with this information
            pass

        m = MultipartEncoder(fields={'field0': 'value0'})
        monitor = MultipartEncoderMonitor(m, callback)
        headers = {'Content-Type': monitor.content_type}
        r = requests.post('https://httpbin.org/post', data=monitor,
                          headers=headers)

    Alternatively, if your use case is very simple, you can use the following
    pattern.

    .. code-block:: python

        from requests_toolbelt import MultipartEncoderMonitor
        import requests

        def callback(monitor):
            # Do something with this information
            pass

        monitor = MultipartEncoderMonitor.from_fields(
            fields={'field0': 'value0'}, callback
            )
        headers = {'Content-Type': montior.content_type}
        r = requests.post('https://httpbin.org/post', data=monitor,
                          headers=headers)

    """

    def __init__(self, encoder, callback=None):
        self.encoder = encoder
        self.callback = callback or IDENTITY
        self.bytes_read = 0
        self.len = self.encoder.len

    @classmethod
    def from_fields(cls, fields, boundary=None, encoding='utf-8', callback=None):
        encoder = MultipartEncoder(fields, boundary, encoding)
        return cls(encoder, callback)

    @property
    def content_type(self):
        return self.encoder.content_type

    def to_string(self):
        return self.read()

    def read(self, size=-1):
        string = self.encoder.read(size)
        self.bytes_read += len(string)
        self.callback(self)
        return string