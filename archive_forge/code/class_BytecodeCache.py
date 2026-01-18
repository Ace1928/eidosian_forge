import errno
import fnmatch
import marshal
import os
import pickle
import stat
import sys
import tempfile
import typing as t
from hashlib import sha1
from io import BytesIO
from types import CodeType
class BytecodeCache:
    """To implement your own bytecode cache you have to subclass this class
    and override :meth:`load_bytecode` and :meth:`dump_bytecode`.  Both of
    these methods are passed a :class:`~jinja2.bccache.Bucket`.

    A very basic bytecode cache that saves the bytecode on the file system::

        from os import path

        class MyCache(BytecodeCache):

            def __init__(self, directory):
                self.directory = directory

            def load_bytecode(self, bucket):
                filename = path.join(self.directory, bucket.key)
                if path.exists(filename):
                    with open(filename, 'rb') as f:
                        bucket.load_bytecode(f)

            def dump_bytecode(self, bucket):
                filename = path.join(self.directory, bucket.key)
                with open(filename, 'wb') as f:
                    bucket.write_bytecode(f)

    A more advanced version of a filesystem based bytecode cache is part of
    Jinja.
    """

    def load_bytecode(self, bucket: Bucket) -> None:
        """Subclasses have to override this method to load bytecode into a
        bucket.  If they are not able to find code in the cache for the
        bucket, it must not do anything.
        """
        raise NotImplementedError()

    def dump_bytecode(self, bucket: Bucket) -> None:
        """Subclasses have to override this method to write the bytecode
        from a bucket back to the cache.  If it unable to do so it must not
        fail silently but raise an exception.
        """
        raise NotImplementedError()

    def clear(self) -> None:
        """Clears the cache.  This method is not used by Jinja but should be
        implemented to allow applications to clear the bytecode cache used
        by a particular environment.
        """

    def get_cache_key(self, name: str, filename: t.Optional[t.Union[str]]=None) -> str:
        """Returns the unique hash key for this template name."""
        hash = sha1(name.encode('utf-8'))
        if filename is not None:
            hash.update(f'|{filename}'.encode())
        return hash.hexdigest()

    def get_source_checksum(self, source: str) -> str:
        """Returns a checksum for the source."""
        return sha1(source.encode('utf-8')).hexdigest()

    def get_bucket(self, environment: 'Environment', name: str, filename: t.Optional[str], source: str) -> Bucket:
        """Return a cache bucket for the given template.  All arguments are
        mandatory but filename may be `None`.
        """
        key = self.get_cache_key(name, filename)
        checksum = self.get_source_checksum(source)
        bucket = Bucket(environment, key, checksum)
        self.load_bytecode(bucket)
        return bucket

    def set_bucket(self, bucket: Bucket) -> None:
        """Put the bucket into the cache."""
        self.dump_bytecode(bucket)