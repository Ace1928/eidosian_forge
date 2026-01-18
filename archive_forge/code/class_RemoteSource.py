from __future__ import absolute_import, print_function, division
import logging
import sys
from contextlib import contextmanager
from petl.compat import PY3
from petl.io.sources import register_reader, register_writer, get_reader, get_writer
class RemoteSource(object):
    """Read or write directly from files in remote filesystems.

    This source handles many filesystems that are selected based on the
    protocol passed in the `url` argument.

    The url should be specified in `to..()` and `from...()` functions. E.g.::

        >>> import petl as etl
        >>>
        >>> def example_s3():
        ...     url = 's3://mybucket/prefix/to/myfilename.csv'
        ...     data = b'foo,bar\\na,1\\nb,2\\nc,2\\n'
        ...
        ...     etl.tocsv(data, url)
        ...     tbl = etl.fromcsv(url)
        ...
        >>> example_s3() # doctest: +SKIP
        +-----+-----+
        | foo | bar |
        +=====+=====+
        | 'a' | '1' |
        +-----+-----+
        | 'b' | '2' |
        +-----+-----+
        | 'c' | '2' |
        +-----+-----+

    This source uses `fsspec`_ to provide the data transfer with the remote
    filesystem. Check the `Built-in Implementations <fs_builtin>`_ for available
    remote implementations.

    Some filesystem can use `URL chaining <fs_chain>`_ for compound I/O.

    .. note::

        For working this source require `fsspec`_ to be installed, e.g.::

            $ pip install fsspec

        Some remote filesystems require aditional packages to be installed.
        Check  `Known Implementations <fs_known>`_ for checking what packages
        need to be installed, e.g.::

            $ pip install s3fs     # AWS S3
            $ pip install gcsfs    # Google Cloud Storage
            $ pip install adlfs    # Azure Blob service
            $ pip install paramiko # SFTP
            $ pip install requests # HTTP, github

    .. versionadded:: 1.6.0

    .. _fsspec: https://filesystem-spec.readthedocs.io/en/latest/
    .. _fs_builtin: https://filesystem-spec.readthedocs.io/en/latest/api.html#built-in-implementations
    .. _fs_known: https://filesystem-spec.readthedocs.io/en/latest/api.html#other-known-implementations
    .. _fs_chain: https://filesystem-spec.readthedocs.io/en/latest/features.html#url-chaining
    """

    def __init__(self, url, **kwargs):
        self.url = url
        self.kwargs = kwargs

    def open_file(self, mode='rb'):
        import fsspec
        fs = fsspec.open(self.url, mode=mode, compression='infer', auto_mkdir=False, **self.kwargs)
        return fs

    @contextmanager
    def open(self, mode='rb'):
        mode2 = mode[:1] + 'b'
        fs = self.open_file(mode=mode2)
        with fs as source:
            yield source