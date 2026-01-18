import os
import math
import sys
from typing import Optional, Union, Callable
import pyglet
from pyglet.customtypes import Buffer
class CodecRegistry:
    """Utility class for handling adding and querying of codecs."""

    def __init__(self):
        self._decoders = []
        self._encoders = []
        self._decoder_extensions = {}
        self._encoder_extensions = {}

    def get_encoders(self, filename=None):
        """Get a list of all encoders. If a `filename` is provided, only
        encoders supporting that extension will be returned. An empty list
        will be return if no encoders for that extension are available.
        """
        if filename:
            root, ext = os.path.splitext(filename)
            extension = ext if ext else root
            return self._encoder_extensions.get(extension.lower(), [])
        return self._encoders

    def get_decoders(self, filename=None):
        """Get a list of all decoders. If a `filename` is provided, only
        decoders supporting that extension will be returned. An empty list
        will be return if no encoders for that extension are available.
        """
        if filename:
            root, ext = os.path.splitext(filename)
            extension = ext if ext else root
            return self._decoder_extensions.get(extension.lower(), [])
        return self._decoders

    def add_decoders(self, module):
        """Add a decoder module.  The module must define `get_decoders`.  Once
        added, the appropriate decoders defined in the codec will be returned by
        CodecRegistry.get_decoders.
        """
        for decoder in module.get_decoders():
            self._decoders.append(decoder)
            for extension in decoder.get_file_extensions():
                if extension not in self._decoder_extensions:
                    self._decoder_extensions[extension] = []
                self._decoder_extensions[extension].append(decoder)

    def add_encoders(self, module):
        """Add an encoder module.  The module must define `get_encoders`.  Once
        added, the appropriate encoders defined in the codec will be returned by
        CodecRegistry.get_encoders.
        """
        for encoder in module.get_encoders():
            self._encoders.append(encoder)
            for extension in encoder.get_file_extensions():
                if extension not in self._encoder_extensions:
                    self._encoder_extensions[extension] = []
                self._encoder_extensions[extension].append(encoder)

    def decode(self, filename, file, **kwargs):
        """Attempt to decode a file, using the available registered decoders.
        Any decoders that match the file extension will be tried first. If no
        decoders match the extension, all decoders will then be tried in order.
        """
        first_exception = None
        for decoder in self.get_decoders(filename):
            try:
                return decoder.decode(filename, file, **kwargs)
            except DecodeException as e:
                if not first_exception:
                    first_exception = e
                if file:
                    file.seek(0)
        for decoder in self.get_decoders():
            try:
                return decoder.decode(filename, file, **kwargs)
            except DecodeException:
                if file:
                    file.seek(0)
        if not first_exception:
            raise DecodeException(f'No decoders available for this file type: {filename}')
        raise first_exception

    def encode(self, media, filename, file=None, **kwargs):
        """Attempt to encode a pyglet object to a specified format. All registered
        encoders that advertise support for the specific file extension will be tried.
        If no encoders are available, an EncodeException will be raised.
        """
        first_exception = None
        for encoder in self.get_encoders(filename):
            try:
                return encoder.encode(media, filename, file, **kwargs)
            except EncodeException as e:
                first_exception = first_exception or e
        if not first_exception:
            raise EncodeException(f"No Encoders are available for this extension: '{filename}'")
        raise first_exception