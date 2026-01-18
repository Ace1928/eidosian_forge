from __future__ import absolute_import, unicode_literals
import io
import os
from ._cffi import (  # type: ignore
class ZstdCompressionDict(object):
    """Represents a computed compression dictionary.

    Instances are obtained by calling :py:func:`train_dictionary` or by
    passing bytes obtained from another source into the constructor.

    Instances can be constructed from bytes:

    >>> dict_data = zstandard.ZstdCompressionDict(data)

    It is possible to construct a dictionary from *any* data. If the data
    doesn't begin with a magic header, it will be treated as a *prefix*
    dictionary. *Prefix* dictionaries allow compression operations to
    reference raw data within the dictionary.

    It is possible to force the use of *prefix* dictionaries or to require
    a dictionary header:

    >>> dict_data = zstandard.ZstdCompressionDict(data, dict_type=zstandard.DICT_TYPE_RAWCONTENT)
    >>> dict_data = zstandard.ZstdCompressionDict(data, dict_type=zstandard.DICT_TYPE_FULLDICT)

    You can see how many bytes are in the dictionary by calling ``len()``:

    >>> dict_data = zstandard.train_dictionary(size, samples)
    >>> dict_size = len(dict_data)  # will not be larger than ``size``

    Once you have a dictionary, you can pass it to the objects performing
    compression and decompression:

    >>> dict_data = zstandard.train_dictionary(131072, samples)
    >>> cctx = zstandard.ZstdCompressor(dict_data=dict_data)
    >>> for source_data in input_data:
    ...     compressed = cctx.compress(source_data)
    ...     # Do something with compressed data.
    ...
    >>> dctx = zstandard.ZstdDecompressor(dict_data=dict_data)
    >>> for compressed_data in input_data:
    ...     buffer = io.BytesIO()
    ...     with dctx.stream_writer(buffer) as decompressor:
    ...         decompressor.write(compressed_data)
    ...         # Do something with raw data in ``buffer``.

    Dictionaries have unique integer IDs. You can retrieve this ID via:

    >>> dict_id = zstandard.dictionary_id(dict_data)

    You can obtain the raw data in the dict (useful for persisting and constructing
    a ``ZstdCompressionDict`` later) via ``as_bytes()``:

    >>> dict_data = zstandard.train_dictionary(size, samples)
    >>> raw_data = dict_data.as_bytes()

    By default, when a ``ZstdCompressionDict`` is *attached* to a
    ``ZstdCompressor``, each ``ZstdCompressor`` performs work to prepare the
    dictionary for use. This is fine if only 1 compression operation is being
    performed or if the ``ZstdCompressor`` is being reused for multiple operations.
    But if multiple ``ZstdCompressor`` instances are being used with the dictionary,
    this can add overhead.

    It is possible to *precompute* the dictionary so it can readily be consumed
    by multiple ``ZstdCompressor`` instances:

    >>> d = zstandard.ZstdCompressionDict(data)
    >>> # Precompute for compression level 3.
    >>> d.precompute_compress(level=3)
    >>> # Precompute with specific compression parameters.
    >>> params = zstandard.ZstdCompressionParameters(...)
    >>> d.precompute_compress(compression_params=params)

    .. note::

       When a dictionary is precomputed, the compression parameters used to
       precompute the dictionary overwrite some of the compression parameters
       specified to ``ZstdCompressor``.

    :param data:
       Dictionary data.
    :param dict_type:
       Type of dictionary. One of the ``DICT_TYPE_*`` constants.
    """

    def __init__(self, data, dict_type=DICT_TYPE_AUTO, k=0, d=0):
        assert isinstance(data, bytes)
        self._data = data
        self.k = k
        self.d = d
        if dict_type not in (DICT_TYPE_AUTO, DICT_TYPE_RAWCONTENT, DICT_TYPE_FULLDICT):
            raise ValueError('invalid dictionary load mode: %d; must use DICT_TYPE_* constants')
        self._dict_type = dict_type
        self._cdict = None

    def __len__(self):
        return len(self._data)

    def dict_id(self):
        """Obtain the integer ID of the dictionary."""
        return int(lib.ZDICT_getDictID(self._data, len(self._data)))

    def as_bytes(self):
        """Obtain the ``bytes`` representation of the dictionary."""
        return self._data

    def precompute_compress(self, level=0, compression_params=None):
        """Precompute a dictionary os it can be used by multiple compressors.

        Calling this method on an instance that will be used by multiple
        :py:class:`ZstdCompressor` instances will improve performance.
        """
        if level and compression_params:
            raise ValueError('must only specify one of level or compression_params')
        if not level and (not compression_params):
            raise ValueError('must specify one of level or compression_params')
        if level:
            cparams = lib.ZSTD_getCParams(level, 0, len(self._data))
        else:
            cparams = ffi.new('ZSTD_compressionParameters')
            cparams.chainLog = compression_params.chain_log
            cparams.hashLog = compression_params.hash_log
            cparams.minMatch = compression_params.min_match
            cparams.searchLog = compression_params.search_log
            cparams.strategy = compression_params.strategy
            cparams.targetLength = compression_params.target_length
            cparams.windowLog = compression_params.window_log
        cdict = lib.ZSTD_createCDict_advanced(self._data, len(self._data), lib.ZSTD_dlm_byRef, self._dict_type, cparams, lib.ZSTD_defaultCMem)
        if cdict == ffi.NULL:
            raise ZstdError('unable to precompute dictionary')
        self._cdict = ffi.gc(cdict, lib.ZSTD_freeCDict, size=lib.ZSTD_sizeof_CDict(cdict))

    @property
    def _ddict(self):
        ddict = lib.ZSTD_createDDict_advanced(self._data, len(self._data), lib.ZSTD_dlm_byRef, self._dict_type, lib.ZSTD_defaultCMem)
        if ddict == ffi.NULL:
            raise ZstdError('could not create decompression dict')
        ddict = ffi.gc(ddict, lib.ZSTD_freeDDict, size=lib.ZSTD_sizeof_DDict(ddict))
        self.__dict__['_ddict'] = ddict
        return ddict