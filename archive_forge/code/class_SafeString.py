import sys, codecs
class SafeString(object):
    """
    A wrapper providing robust conversion to `str` and `unicode`.
    """

    def __init__(self, data, encoding=None, encoding_errors='backslashreplace', decoding_errors='replace'):
        self.data = data
        self.encoding = encoding or getattr(data, 'encoding', None) or locale_encoding or 'ascii'
        self.encoding_errors = encoding_errors
        self.decoding_errors = decoding_errors

    def __str__(self):
        try:
            return str(self.data)
        except UnicodeEncodeError:
            if isinstance(self.data, Exception):
                args = [str(SafeString(arg, self.encoding, self.encoding_errors)) for arg in self.data.args]
                return ', '.join(args)
            if isinstance(self.data, str):
                if sys.version_info > (3, 0):
                    return self.data
                else:
                    return self.data.encode(self.encoding, self.encoding_errors)
            raise

    def __unicode__(self):
        """
        Return unicode representation of `self.data`.

        Try ``unicode(self.data)``, catch `UnicodeError` and

        * if `self.data` is an Exception instance, work around
          http://bugs.python.org/issue2517 with an emulation of
          Exception.__unicode__,

        * else decode with `self.encoding` and `self.decoding_errors`.
        """
        try:
            u = str(self.data)
            if isinstance(self.data, EnvironmentError):
                u = u.replace(": u'", ": '")
            return u
        except UnicodeError as error:
            if isinstance(self.data, EnvironmentError):
                return "[Errno %s] %s: '%s'" % (self.data.errno, SafeString(self.data.strerror, self.encoding, self.decoding_errors), SafeString(self.data.filename, self.encoding, self.decoding_errors))
            if isinstance(self.data, Exception):
                args = [str(SafeString(arg, self.encoding, decoding_errors=self.decoding_errors)) for arg in self.data.args]
                return ', '.join(args)
            if isinstance(error, UnicodeDecodeError):
                return str(self.data, self.encoding, self.decoding_errors)
            raise