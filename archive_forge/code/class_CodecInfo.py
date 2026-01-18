import builtins
import sys
class CodecInfo(tuple):
    """Codec details when looking up the codec registry"""
    _is_text_encoding = True

    def __new__(cls, encode, decode, streamreader=None, streamwriter=None, incrementalencoder=None, incrementaldecoder=None, name=None, *, _is_text_encoding=None):
        self = tuple.__new__(cls, (encode, decode, streamreader, streamwriter))
        self.name = name
        self.encode = encode
        self.decode = decode
        self.incrementalencoder = incrementalencoder
        self.incrementaldecoder = incrementaldecoder
        self.streamwriter = streamwriter
        self.streamreader = streamreader
        if _is_text_encoding is not None:
            self._is_text_encoding = _is_text_encoding
        return self

    def __repr__(self):
        return '<%s.%s object for encoding %s at %#x>' % (self.__class__.__module__, self.__class__.__qualname__, self.name, id(self))