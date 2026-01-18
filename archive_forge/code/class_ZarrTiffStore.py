from __future__ import annotations
import binascii
import collections
import datetime
import enum
import glob
import io
import json
import logging
import math
import os
import re
import struct
import sys
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from functools import cached_property
import numpy
from typing import TYPE_CHECKING, BinaryIO, cast, final, overload
@final
class ZarrTiffStore(ZarrStore):
    """Zarr store interface to image array in TiffPage or TiffPageSeries.

    ZarrTiffStore is using a TiffFile instance for reading and decoding chunks.
    Therefore, ZarrTiffStore instances cannot be pickled.

    For writing, image data must be stored in uncompressed, unpredicted,
    and unpacked form. Sparse strips and tiles are not written.

    Parameters:
        arg:
            TIFF page or series to wrap as Zarr store.
        level:
            Pyramidal level to wrap. The default is 0.
        chunkmode:
            Use strips or tiles (0) or whole page data (2) as chunks.
            The default is 0.
        fillvalue:
            Value to use for missing chunks. The default is 0.
        zattrs:
            Additional attributes to store in `.zattrs`.
        multiscales:
            Create a multiscales compatible Zarr group store.
            By default, create a Zarr array store for pages and non-pyramidal
            series.
        lock:
            Reentrant lock to synchronize seeks and reads from file.
            By default, the lock of the parent's file handle is used.
        squeeze:
            Remove length-1 dimensions from shape of TiffPageSeries.
        maxworkers:
            Maximum number of threads to concurrently decode strips or tiles
            if `chunkmode=2`.
            If *None* or *0*, use up to :py:attr:`_TIFF.MAXWORKERS` threads.
        buffersize:
            Approximate number of bytes to read from file in one pass
            if `chunkmode=2`. The default is :py:attr:`_TIFF.BUFFERSIZE`.
        _openfiles:
            Internal API.

    """
    _data: list[TiffPageSeries]
    _filecache: FileCache
    _transform: Callable[[NDArray[Any]], NDArray[Any]] | None
    _maxworkers: int | None
    _buffersize: int | None
    _squeeze: bool | None
    _writable: bool
    _multiscales: bool

    def __init__(self, arg: TiffPage | TiffFrame | TiffPageSeries, /, *, level: int | None=None, chunkmode: CHUNKMODE | int | str | None=None, fillvalue: int | float | None=None, zattrs: dict[str, Any] | None=None, multiscales: bool | None=None, lock: threading.RLock | NullContext | None=None, squeeze: bool | None=None, maxworkers: int | None=None, buffersize: int | None=None, _openfiles: int | None=None) -> None:
        super().__init__(fillvalue=fillvalue, chunkmode=chunkmode)
        if self._chunkmode not in {0, 2}:
            raise NotImplementedError(f'{self._chunkmode!r} not implemented')
        self._squeeze = None if squeeze is None else bool(squeeze)
        self._maxworkers = maxworkers
        self._buffersize = buffersize
        if isinstance(arg, TiffPageSeries):
            self._data = arg.levels
            self._transform = arg.transform
            if multiscales is not None and (not multiscales):
                level = 0
            if level is not None:
                self._data = [self._data[level]]
            name = arg.name
        else:
            self._data = [TiffPageSeries([arg])]
            self._transform = None
            name = 'Unnamed'
        fh = self._data[0].keyframe.parent._parent.filehandle
        self._writable = fh.writable() and self._chunkmode == 0
        if lock is None:
            fh.set_lock(True)
            lock = fh.lock
        self._filecache = FileCache(size=_openfiles, lock=lock)
        zattrs = {} if zattrs is None else dict(zattrs)
        if multiscales or len(self._data) > 1:
            self._multiscales = True
            if '_ARRAY_DIMENSIONS' in zattrs:
                array_dimensions = zattrs.pop('_ARRAY_DIMENSIONS')
            else:
                array_dimensions = list(self._data[0].get_axes(squeeze))
            self._store['.zgroup'] = ZarrStore._json({'zarr_format': 2})
            self._store['.zattrs'] = ZarrStore._json({'multiscales': [{'version': '0.1', 'name': name, 'datasets': [{'path': str(i)} for i in range(len(self._data))], 'metadata': {}}], **zattrs})
            shape0 = self._data[0].get_shape(squeeze)
            for level, series in enumerate(self._data):
                keyframe = series.keyframe
                keyframe.decode
                shape = series.get_shape(squeeze)
                dtype = series.dtype
                if fillvalue is None:
                    self._fillvalue = fillvalue = keyframe.nodata
                if self._chunkmode:
                    chunks = keyframe.shape
                else:
                    chunks = keyframe.chunks
                self._store[f'{level}/.zattrs'] = ZarrStore._json({'_ARRAY_DIMENSIONS': [f'{ax}{level}' if i != j else ax for ax, i, j in zip(array_dimensions, shape, shape0)]})
                self._store[f'{level}/.zarray'] = ZarrStore._json({'zarr_format': 2, 'shape': shape, 'chunks': ZarrTiffStore._chunks(chunks, shape), 'dtype': ZarrStore._dtype_str(dtype), 'compressor': None, 'fill_value': ZarrStore._value(fillvalue, dtype), 'order': 'C', 'filters': None})
                if self._writable:
                    self._writable = ZarrTiffStore._is_writable(keyframe)
        else:
            self._multiscales = False
            series = self._data[0]
            keyframe = series.keyframe
            keyframe.decode
            shape = series.get_shape(squeeze)
            dtype = series.dtype
            if fillvalue is None:
                self._fillvalue = fillvalue = keyframe.nodata
            if self._chunkmode:
                chunks = keyframe.shape
            else:
                chunks = keyframe.chunks
            if '_ARRAY_DIMENSIONS' not in zattrs:
                zattrs['_ARRAY_DIMENSIONS'] = list(series.get_axes(squeeze))
            self._store['.zattrs'] = ZarrStore._json(zattrs)
            self._store['.zarray'] = ZarrStore._json({'zarr_format': 2, 'shape': shape, 'chunks': ZarrTiffStore._chunks(chunks, shape), 'dtype': ZarrStore._dtype_str(dtype), 'compressor': None, 'fill_value': ZarrStore._value(fillvalue, dtype), 'order': 'C', 'filters': None})
            if self._writable:
                self._writable = ZarrTiffStore._is_writable(keyframe)

    def close(self) -> None:
        """Close open file handles."""
        if hasattr(self, '_filecache'):
            self._filecache.clear()

    def write_fsspec(self, jsonfile: str | os.PathLike[Any] | TextIO, /, url: str, *, groupname: str | None=None, templatename: str | None=None, compressors: dict[COMPRESSION | int, str | None] | None=None, version: int | None=None, _shape: Sequence[int] | None=None, _axes: Sequence[str] | None=None, _index: Sequence[int] | None=None, _append: bool=False, _close: bool=True) -> None:
        """Write fsspec ReferenceFileSystem as JSON to file.

        Parameters:
            jsonfile:
                Name or open file handle of output JSON file.
            url:
                Remote location of TIFF file(s) without file name(s).
            groupname:
                Zarr group name.
            templatename:
                Version 1 URL template name. The default is 'u'.
            compressors:
                Mapping of :py:class:`COMPRESSION` codes to Numcodecs codec
                names.
            version:
                Version of fsspec file to write. The default is 0.
            _shape:
                Shape of file sequence (experimental).
            _axes:
                Axes of file sequence (experimental).
            _index
                Index of file in sequence (experimental).
            _append:
                If *True*, only write index keys and values (experimental).
            _close:
                If *True*, no more appends (experimental).

        Raises:
            ValueError:
                ZarrTiffStore cannot be represented as ReferenceFileSystem
                due to features that are not supported by Zarr, Numcodecs,
                or Imagecodecs:

                - compressors, such as CCITT
                - filters, such as bitorder reversal, packed integers
                - dtypes, such as float24
                - JPEGTables in multi-page files
                - incomplete chunks, such as `imagelength % rowsperstrip != 0`

                Files containing incomplete tiles may fail at runtime.

        Notes:
            Parameters `_shape`,  `_axes`, `_index`, `_append`, and `_close`
            are an experimental API for joining the ReferenceFileSystems of
            multiple files of a TiffSequence.

        References:
            - `fsspec ReferenceFileSystem format
              <https://github.com/fsspec/kerchunk>`_

        """
        compressors = {1: None, 8: 'zlib', 32946: 'zlib', 34925: 'lzma', 50013: 'zlib', 5: 'imagecodecs_lzw', 7: 'imagecodecs_jpeg', 22610: 'imagecodecs_jpegxr', 32773: 'imagecodecs_packbits', 33003: 'imagecodecs_jpeg2k', 33004: 'imagecodecs_jpeg2k', 33005: 'imagecodecs_jpeg2k', 33007: 'imagecodecs_jpeg', 34712: 'imagecodecs_jpeg2k', 34887: 'imagecodecs_lerc', 34892: 'imagecodecs_jpeg', 34933: 'imagecodecs_png', 34934: 'imagecodecs_jpegxr', 48124: 'imagecodecs_jetraw', 50000: 'imagecodecs_zstd', 50001: 'imagecodecs_webp', 50002: 'imagecodecs_jpegxl', **({} if compressors is None else compressors)}
        for series in self._data:
            errormsg = ' not supported by the fsspec ReferenceFileSystem'
            keyframe = series.keyframe
            if keyframe.compression in {65000, 65001, 65002} and keyframe.parent.is_eer:
                compressors[keyframe.compression] = 'imagecodecs_eer'
            if keyframe.compression not in compressors:
                raise ValueError(f'{keyframe.compression!r} is' + errormsg)
            if keyframe.fillorder != 1:
                raise ValueError(f'{keyframe.fillorder!r} is' + errormsg)
            if keyframe.sampleformat not in {1, 2, 3, 6}:
                raise ValueError(f'{keyframe.sampleformat!r} is' + errormsg)
            if keyframe.bitspersample not in {8, 16, 32, 64, 128} and keyframe.compression not in {7, 33007, 34892} and (compressors[keyframe.compression] != 'imagecodecs_eer'):
                raise ValueError(f'BitsPerSample {keyframe.bitspersample} is' + errormsg)
            if not self._chunkmode and (not keyframe.is_tiled) and keyframe.imagelength % keyframe.rowsperstrip:
                raise ValueError('incomplete chunks are' + errormsg)
            if self._chunkmode and (not keyframe.is_final):
                raise ValueError(f'{self._chunkmode!r} is' + errormsg)
            if keyframe.jpegtables is not None and len(series.pages) > 1:
                raise ValueError('JPEGTables in multi-page files are' + errormsg)
        if url is None:
            url = ''
        elif url and url[-1] != '/':
            url += '/'
        url = url.replace('\\', '/')
        if groupname is None:
            groupname = ''
        elif groupname and groupname[-1] != '/':
            groupname += '/'
        byteorder: ByteOrder | None = '<' if sys.byteorder == 'big' else '>'
        if self._data[0].keyframe.parent.byteorder != byteorder or self._data[0].keyframe.dtype is None or self._data[0].keyframe.dtype.itemsize == 1:
            byteorder = None
        index: str
        _shape = [] if _shape is None else list(_shape)
        _axes = [] if _axes is None else list(_axes)
        if len(_shape) != len(_axes):
            raise ValueError('len(_shape) != len(_axes)')
        if _index is None:
            index = ''
        elif len(_shape) != len(_index):
            raise ValueError('len(_shape) != len(_index)')
        elif _index:
            index = '.'.join((str(i) for i in _index))
            index += '.'
        refs: dict[str, Any] = {}
        refzarr: dict[str, Any]
        if version == 1:
            if _append:
                raise ValueError('cannot append to version 1')
            if templatename is None:
                templatename = 'u'
            refs['version'] = 1
            refs['templates'] = {}
            refs['gen'] = []
            templates = {}
            if self._data[0].is_multifile:
                i = 0
                for page in self._data[0].pages:
                    if page is None or page.keyframe is None:
                        continue
                    fname = page.keyframe.parent.filehandle.name
                    if fname in templates:
                        continue
                    key = f'{templatename}{i}'
                    templates[fname] = '{{%s}}' % key
                    refs['templates'][key] = url + fname
                    i += 1
            else:
                fname = self._data[0].keyframe.parent.filehandle.name
                key = f'{templatename}'
                templates[fname] = '{{%s}}' % key
                refs['templates'][key] = url + fname
            refs['refs'] = refzarr = {}
        else:
            refzarr = refs
        if not _append:
            if groupname:
                refzarr['.zgroup'] = ZarrStore._json({'zarr_format': 2}).decode()
            for key, value in self._store.items():
                if '.zattrs' in key and _axes:
                    value = json.loads(value)
                    if '_ARRAY_DIMENSIONS' in value:
                        value['_ARRAY_DIMENSIONS'] = _axes + value['_ARRAY_DIMENSIONS']
                    value = ZarrStore._json(value)
                elif '.zarray' in key:
                    level = int(key.split('/')[0]) if '/' in key else 0
                    keyframe = self._data[level].keyframe
                    value = json.loads(value)
                    if _shape:
                        value['shape'] = _shape + value['shape']
                        value['chunks'] = [1] * len(_shape) + value['chunks']
                    codec_id = compressors[keyframe.compression]
                    if codec_id == 'imagecodecs_jpeg':
                        jpegtables = keyframe.jpegtables
                        if jpegtables is None:
                            tables = None
                        else:
                            import base64
                            tables = base64.b64encode(jpegtables).decode()
                        jpegheader = keyframe.jpegheader
                        if jpegheader is None:
                            header = None
                        else:
                            import base64
                            header = base64.b64encode(jpegheader).decode()
                        colorspace_jpeg, colorspace_data = jpeg_decode_colorspace(keyframe.photometric, keyframe.planarconfig, keyframe.extrasamples, keyframe.is_jfif)
                        value['compressor'] = {'id': codec_id, 'tables': tables, 'header': header, 'bitspersample': keyframe.bitspersample, 'colorspace_jpeg': colorspace_jpeg, 'colorspace_data': colorspace_data}
                    elif codec_id == 'imagecodecs_webp' and keyframe.samplesperpixel == 4:
                        value['compressor'] = {'id': codec_id, 'hasalpha': True}
                    elif codec_id == 'imagecodecs_eer':
                        if keyframe.compression == 65002:
                            rlebits = int(keyframe.tags.valueof(65007, 7))
                            horzbits = int(keyframe.tags.valueof(65008, 2))
                            vertbits = int(keyframe.tags.valueof(65009, 2))
                        elif keyframe.compression == 65001:
                            rlebits = 7
                            horzbits = 2
                            vertbits = 2
                        else:
                            rlebits = 8
                            horzbits = 2
                            vertbits = 2
                        value['compressor'] = {'id': codec_id, 'shape': keyframe.chunks, 'rlebits': rlebits, 'horzbits': horzbits, 'vertbits': vertbits}
                    elif codec_id is not None:
                        value['compressor'] = {'id': codec_id}
                    if byteorder is not None:
                        value['dtype'] = byteorder + value['dtype'][1:]
                    if keyframe.predictor > 1:
                        if keyframe.predictor in {2, 34892, 34893}:
                            filter_id = 'imagecodecs_delta'
                        else:
                            filter_id = 'imagecodecs_floatpred'
                        if keyframe.predictor <= 3:
                            dist = 1
                        elif keyframe.predictor in {34892, 34894}:
                            dist = 2
                        else:
                            dist = 4
                        if keyframe.planarconfig == 1 and keyframe.samplesperpixel > 1:
                            axis = -2
                        else:
                            axis = -1
                        value['filters'] = [{'id': filter_id, 'axis': axis, 'dist': dist, 'shape': value['chunks'], 'dtype': value['dtype']}]
                    value = ZarrStore._json(value)
                refzarr[groupname + key] = value.decode()
        fh: TextIO
        if hasattr(jsonfile, 'write'):
            fh = jsonfile
        else:
            fh = open(jsonfile, 'w', encoding='utf-8')
        if version == 1:
            fh.write(json.dumps(refs, indent=1).rsplit('}"', 1)[0] + '}"')
            indent = '  '
        elif _append:
            indent = ' '
        else:
            fh.write(json.dumps(refs, indent=1)[:-2])
            indent = ' '
        for key, value in self._store.items():
            if '.zarray' in key:
                value = json.loads(value)
                shape = value['shape']
                chunks = value['chunks']
                levelstr = key.split('/')[0] + '/' if '/' in key else ''
                for chunkindex in ZarrStore._ndindex(shape, chunks):
                    key = levelstr + chunkindex
                    keyframe, page, _, offset, bytecount = self._parse_key(key)
                    key = levelstr + index + chunkindex
                    if page and self._chunkmode and (offset is None):
                        offset = page.dataoffsets[0]
                        bytecount = keyframe.nbytes
                    if offset and bytecount:
                        fname = keyframe.parent.filehandle.name
                        if version == 1:
                            fname = templates[fname]
                        else:
                            fname = f'{url}{fname}'
                        fh.write(f',\n{indent}"{groupname}{key}": ["{fname}", {offset}, {bytecount}]')
        if version == 1:
            fh.write('\n }\n}')
        elif _close:
            fh.write('\n}')
        if not hasattr(jsonfile, 'write'):
            fh.close()

    def _contains(self, key: str, /) -> bool:
        """Return if key is in store."""
        try:
            _, page, _, offset, bytecount = self._parse_key(key)
        except (KeyError, IndexError):
            return False
        if self._chunkmode and offset is None:
            return True
        return page is not None and offset is not None and (bytecount is not None) and (offset > 0) and (bytecount > 0)

    def _getitem(self, key: str, /) -> NDArray[Any]:
        """Return chunk from file."""
        keyframe, page, chunkindex, offset, bytecount = self._parse_key(key)
        if page is None or offset == 0 or bytecount == 0:
            raise KeyError(key)
        fh = page.parent.filehandle
        if self._chunkmode and offset is None:
            self._filecache.open(fh)
            chunk = page.asarray(lock=self._filecache.lock, maxworkers=self._maxworkers, buffersize=self._buffersize)
            self._filecache.close(fh)
            if self._transform is not None:
                chunk = self._transform(chunk)
            return chunk
        assert offset is not None and bytecount is not None
        chunk_bytes = self._filecache.read(fh, offset, bytecount)
        decodeargs: dict[str, Any] = {'_fullsize': True}
        if page.jpegtables is not None:
            decodeargs['jpegtables'] = page.jpegtables
        if keyframe.jpegheader is not None:
            decodeargs['jpegheader'] = keyframe.jpegheader
        assert chunkindex is not None
        chunk = keyframe.decode(chunk_bytes, chunkindex, **decodeargs)[0]
        assert chunk is not None
        if self._transform is not None:
            chunk = self._transform(chunk)
        if self._chunkmode:
            chunks = keyframe.shape
        else:
            chunks = keyframe.chunks
        if chunk.size != product(chunks):
            raise RuntimeError(f'{chunk.size} != {product(chunks)}')
        return chunk

    def _setitem(self, key: str, value: bytes, /) -> None:
        """Write chunk to file."""
        if not self._writable:
            raise PermissionError('ZarrStore is read-only')
        keyframe, page, chunkindex, offset, bytecount = self._parse_key(key)
        if page is None or offset is None or offset == 0 or (bytecount is None) or (bytecount == 0):
            return
        if bytecount < len(value):
            value = value[:bytecount]
        self._filecache.write(page.parent.filehandle, offset, value)

    def _parse_key(self, key: str, /) -> tuple[TiffPage, TiffPage | TiffFrame | None, int | None, int | None, int | None]:
        """Return keyframe, page, index, offset, and bytecount from key.

        Raise KeyError if key is not valid.

        """
        if self._multiscales:
            try:
                level, key = key.split('/')
                series = self._data[int(level)]
            except (ValueError, IndexError) as exc:
                raise KeyError(key) from exc
        else:
            series = self._data[0]
        keyframe = series.keyframe
        pageindex, chunkindex = self._indices(key, series)
        if pageindex > 0 and len(series) == 1:
            if series.dataoffset is None:
                raise RuntimeError('truncated series is not contiguous')
            page = series[0]
            if page is None or page.dtype is None or page.keyframe is None:
                return (keyframe, None, chunkindex, 0, 0)
            offset = pageindex * page.size * page.dtype.itemsize
            try:
                offset += page.dataoffsets[chunkindex]
            except IndexError as exc:
                raise KeyError(key) from exc
            if self._chunkmode:
                bytecount = page.size * page.dtype.itemsize
                return (page.keyframe, page, chunkindex, offset, bytecount)
        elif self._chunkmode:
            with self._filecache.lock:
                page = series[pageindex]
            if page is None or page.keyframe is None:
                return (keyframe, None, None, 0, 0)
            return (page.keyframe, page, None, None, None)
        else:
            with self._filecache.lock:
                page = series[pageindex]
            if page is None or page.keyframe is None:
                return (keyframe, None, chunkindex, 0, 0)
            try:
                offset = page.dataoffsets[chunkindex]
            except IndexError:
                return (page.keyframe, page, chunkindex, 0, 0)
        try:
            bytecount = page.databytecounts[chunkindex]
        except IndexError as exc:
            raise KeyError(key) from exc
        return (page.keyframe, page, chunkindex, offset, bytecount)

    def _indices(self, key: str, series: TiffPageSeries, /) -> tuple[int, int]:
        """Return page and strile indices from Zarr chunk index."""
        keyframe = series.keyframe
        shape = series.get_shape(self._squeeze)
        try:
            indices = [int(i) for i in key.split('.')]
        except ValueError as exc:
            raise KeyError(key) from exc
        assert len(indices) == len(shape)
        if self._chunkmode:
            chunked = (1,) * len(keyframe.shape)
        else:
            chunked = keyframe.chunked
        p = 1
        for i, s in enumerate(shape[::-1]):
            p *= s
            if p == keyframe.size:
                i = len(indices) - i - 1
                frames_indices = indices[:i]
                strile_indices = indices[i:]
                frames_chunked = shape[:i]
                strile_chunked = list(shape[i:])
                break
        else:
            raise RuntimeError
        if len(strile_chunked) == len(keyframe.shape):
            strile_chunked = list(chunked)
        else:
            i = len(strile_indices) - 1
            j = len(keyframe.shape) - 1
            while True:
                if strile_chunked[i] == keyframe.shape[j]:
                    strile_chunked[i] = chunked[j]
                    i -= 1
                    j -= 1
                elif strile_chunked[i] == 1:
                    i -= 1
                else:
                    raise RuntimeError('shape does not match page shape')
                if i < 0 or j < 0:
                    break
            assert product(strile_chunked) == product(chunked)
        if len(frames_indices) > 0:
            frameindex = int(numpy.ravel_multi_index(frames_indices, frames_chunked))
        else:
            frameindex = 0
        if len(strile_indices) > 0:
            strileindex = int(numpy.ravel_multi_index(strile_indices, strile_chunked))
        else:
            strileindex = 0
        return (frameindex, strileindex)

    @staticmethod
    def _chunks(chunks: tuple[int, ...], shape: tuple[int, ...], /) -> tuple[int, ...]:
        """Return chunks with same length as shape."""
        ndim = len(shape)
        if ndim == 0:
            return ()
        if 0 in shape:
            return (1,) * ndim
        newchunks = []
        i = ndim - 1
        j = len(chunks) - 1
        while True:
            if j < 0:
                newchunks.append(1)
                i -= 1
            elif shape[i] > 1 and chunks[j] > 1:
                newchunks.append(chunks[j])
                i -= 1
                j -= 1
            elif shape[i] == chunks[j]:
                newchunks.append(1)
                i -= 1
                j -= 1
            elif shape[i] == 1:
                newchunks.append(1)
                i -= 1
            elif chunks[j] == 1:
                newchunks.append(1)
                j -= 1
            else:
                raise RuntimeError
            if i < 0 or ndim == len(newchunks):
                break
        return tuple(newchunks[::-1])

    @staticmethod
    def _is_writable(keyframe: TiffPage) -> bool:
        """Return True if chunks are writable."""
        return keyframe.compression == 1 and keyframe.fillorder == 1 and (keyframe.sampleformat in {1, 2, 3, 6}) and (keyframe.bitspersample in {8, 16, 32, 64, 128})

    def __enter__(self) -> ZarrTiffStore:
        return self

    def __repr__(self) -> str:
        return f'<tifffile.ZarrTiffStore @0x{id(self):016X}>'