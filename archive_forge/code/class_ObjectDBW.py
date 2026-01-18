from gitdb.util import (
from gitdb.utils.encoding import force_text
from gitdb.exc import (
from itertools import chain
from functools import reduce
class ObjectDBW:
    """Defines an interface to create objects in the database"""

    def __init__(self, *args, **kwargs):
        self._ostream = None

    def set_ostream(self, stream):
        """
        Adjusts the stream to which all data should be sent when storing new objects

        :param stream: if not None, the stream to use, if None the default stream
            will be used.
        :return: previously installed stream, or None if there was no override
        :raise TypeError: if the stream doesn't have the supported functionality"""
        cstream = self._ostream
        self._ostream = stream
        return cstream

    def ostream(self):
        """
        Return the output stream

        :return: overridden output stream this instance will write to, or None
            if it will write to the default stream"""
        return self._ostream

    def store(self, istream):
        """
        Create a new object in the database
        :return: the input istream object with its sha set to its corresponding value

        :param istream: IStream compatible instance. If its sha is already set
            to a value, the object will just be stored in the our database format,
            in which case the input stream is expected to be in object format ( header + contents ).
        :raise IOError: if data could not be written"""
        raise NotImplementedError('To be implemented in subclass')