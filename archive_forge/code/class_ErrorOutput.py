import sys, codecs
class ErrorOutput(object):
    """
    Wrapper class for file-like error streams with
    failsave de- and encoding of `str`, `bytes`, `unicode` and
    `Exception` instances.
    """

    def __init__(self, stream=None, encoding=None, encoding_errors='backslashreplace', decoding_errors='replace'):
        """
        :Parameters:
            - `stream`: a file-like object,
                        a string (path to a file),
                        `None` (write to `sys.stderr`, default), or
                        evaluating to `False` (write() requests are ignored).
            - `encoding`: `stream` text encoding. Guessed if None.
            - `encoding_errors`: how to treat encoding errors.
        """
        if stream is None:
            stream = sys.stderr
        elif not stream:
            stream = False
        elif isinstance(stream, str):
            stream = open(stream, 'w')
        elif isinstance(stream, str):
            stream = open(stream.encode(sys.getfilesystemencoding()), 'w')
        self.stream = stream
        'Where warning output is sent.'
        self.encoding = encoding or getattr(stream, 'encoding', None) or locale_encoding or 'ascii'
        'The output character encoding.'
        self.encoding_errors = encoding_errors
        'Encoding error handler.'
        self.decoding_errors = decoding_errors
        'Decoding error handler.'

    def write(self, data):
        """
        Write `data` to self.stream. Ignore, if self.stream is False.

        `data` can be a `string`, `unicode`, or `Exception` instance.
        """
        if self.stream is False:
            return
        if isinstance(data, Exception):
            data = str(SafeString(data, self.encoding, self.encoding_errors, self.decoding_errors))
        try:
            self.stream.write(data)
        except UnicodeEncodeError:
            self.stream.write(data.encode(self.encoding, self.encoding_errors))
        except TypeError:
            if isinstance(data, str):
                self.stream.write(data.encode(self.encoding, self.encoding_errors))
                return
            if self.stream in (sys.stderr, sys.stdout):
                self.stream.buffer.write(data)
            else:
                self.stream.write(str(data, self.encoding, self.decoding_errors))

    def close(self):
        """
        Close the error-output stream.

        Ignored if the stream is` sys.stderr` or `sys.stdout` or has no
        close() method.
        """
        if self.stream in (sys.stdout, sys.stderr):
            return
        try:
            self.stream.close()
        except AttributeError:
            pass