import os as _os
import sys as _sys
from os import SEEK_SET, SEEK_CUR, SEEK_END
from ctypes.util import find_library as _find_library
from _soundfile import ffi as _ffi
class SoundFile(object):
    """A sound file.

    For more documentation see the __init__() docstring (which is also
    used for the online documentation (https://python-soundfile.readthedocs.io/).

    """

    def __init__(self, file, mode='r', samplerate=None, channels=None, subtype=None, endian=None, format=None, closefd=True):
        """Open a sound file.

        If a file is opened with `mode` ``'r'`` (the default) or
        ``'r+'``, no sample rate, channels or file format need to be
        given because the information is obtained from the file. An
        exception is the ``'RAW'`` data format, which always requires
        these data points.

        File formats consist of three case-insensitive strings:

        * a *major format* which is by default obtained from the
          extension of the file name (if known) and which can be
          forced with the format argument (e.g. ``format='WAVEX'``).
        * a *subtype*, e.g. ``'PCM_24'``. Most major formats have a
          default subtype which is used if no subtype is specified.
        * an *endian-ness*, which doesn't have to be specified at all in
          most cases.

        A `SoundFile` object is a *context manager*, which means
        if used in a "with" statement, `close()` is automatically
        called when reaching the end of the code block inside the "with"
        statement.

        Parameters
        ----------
        file : str or int or file-like object
            The file to open.  This can be a file name, a file
            descriptor or a Python file object (or a similar object with
            the methods ``read()``/``readinto()``, ``write()``,
            ``seek()`` and ``tell()``).
        mode : {'r', 'r+', 'w', 'w+', 'x', 'x+'}, optional
            Open mode.  Has to begin with one of these three characters:
            ``'r'`` for reading, ``'w'`` for writing (truncates *file*)
            or ``'x'`` for writing (raises an error if *file* already
            exists).  Additionally, it may contain ``'+'`` to open
            *file* for both reading and writing.
            The character ``'b'`` for *binary mode* is implied because
            all sound files have to be opened in this mode.
            If *file* is a file descriptor or a file-like object,
            ``'w'`` doesn't truncate and ``'x'`` doesn't raise an error.
        samplerate : int
            The sample rate of the file.  If `mode` contains ``'r'``,
            this is obtained from the file (except for ``'RAW'`` files).
        channels : int
            The number of channels of the file.
            If `mode` contains ``'r'``, this is obtained from the file
            (except for ``'RAW'`` files).
        subtype : str, sometimes optional
            The subtype of the sound file.  If `mode` contains ``'r'``,
            this is obtained from the file (except for ``'RAW'``
            files), if not, the default value depends on the selected
            `format` (see `default_subtype()`).
            See `available_subtypes()` for all possible subtypes for
            a given `format`.
        endian : {'FILE', 'LITTLE', 'BIG', 'CPU'}, sometimes optional
            The endian-ness of the sound file.  If `mode` contains
            ``'r'``, this is obtained from the file (except for
            ``'RAW'`` files), if not, the default value is ``'FILE'``,
            which is correct in most cases.
        format : str, sometimes optional
            The major format of the sound file.  If `mode` contains
            ``'r'``, this is obtained from the file (except for
            ``'RAW'`` files), if not, the default value is determined
            from the file extension.  See `available_formats()` for
            all possible values.
        closefd : bool, optional
            Whether to close the file descriptor on `close()`. Only
            applicable if the *file* argument is a file descriptor.

        Examples
        --------
        >>> from soundfile import SoundFile

        Open an existing file for reading:

        >>> myfile = SoundFile('existing_file.wav')
        >>> # do something with myfile
        >>> myfile.close()

        Create a new sound file for reading and writing using a with
        statement:

        >>> with SoundFile('new_file.wav', 'x+', 44100, 2) as myfile:
        >>>     # do something with myfile
        >>>     # ...
        >>>     assert not myfile.closed
        >>>     # myfile.close() is called automatically at the end
        >>> assert myfile.closed

        """
        file = file.__fspath__() if hasattr(file, '__fspath__') else file
        self._name = file
        if mode is None:
            mode = getattr(file, 'mode', None)
        mode_int = _check_mode(mode)
        self._mode = mode
        self._info = _create_info_struct(file, mode, samplerate, channels, format, subtype, endian)
        self._file = self._open(file, mode_int, closefd)
        if set(mode).issuperset('r+') and self.seekable():
            self.seek(0)
        _snd.sf_command(self._file, _snd.SFC_SET_CLIPPING, _ffi.NULL, _snd.SF_TRUE)
    name = property(lambda self: self._name)
    'The file name of the sound file.'
    mode = property(lambda self: self._mode)
    'The open mode the sound file was opened with.'
    samplerate = property(lambda self: self._info.samplerate)
    'The sample rate of the sound file.'
    frames = property(lambda self: self._info.frames)
    'The number of frames in the sound file.'
    channels = property(lambda self: self._info.channels)
    'The number of channels in the sound file.'
    format = property(lambda self: _format_str(self._info.format & _snd.SF_FORMAT_TYPEMASK))
    'The major format of the sound file.'
    subtype = property(lambda self: _format_str(self._info.format & _snd.SF_FORMAT_SUBMASK))
    'The subtype of data in the the sound file.'
    endian = property(lambda self: _format_str(self._info.format & _snd.SF_FORMAT_ENDMASK))
    'The endian-ness of the data in the sound file.'
    format_info = property(lambda self: _format_info(self._info.format & _snd.SF_FORMAT_TYPEMASK)[1])
    'A description of the major format of the sound file.'
    subtype_info = property(lambda self: _format_info(self._info.format & _snd.SF_FORMAT_SUBMASK)[1])
    'A description of the subtype of the sound file.'
    sections = property(lambda self: self._info.sections)
    'The number of sections of the sound file.'
    closed = property(lambda self: self._file is None)
    'Whether the sound file is closed or not.'
    _errorcode = property(lambda self: _snd.sf_error(self._file))
    'A pending sndfile error code.'

    @property
    def extra_info(self):
        """Retrieve the log string generated when opening the file."""
        info = _ffi.new('char[]', 2 ** 14)
        _snd.sf_command(self._file, _snd.SFC_GET_LOG_INFO, info, _ffi.sizeof(info))
        return _ffi.string(info).decode('utf-8', 'replace')
    _file = None

    def __repr__(self):
        return 'SoundFile({0.name!r}, mode={0.mode!r}, samplerate={0.samplerate}, channels={0.channels}, format={0.format!r}, subtype={0.subtype!r}, endian={0.endian!r})'.format(self)

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __setattr__(self, name, value):
        """Write text meta-data in the sound file through properties."""
        if name in _str_types:
            self._check_if_closed()
            err = _snd.sf_set_string(self._file, _str_types[name], value.encode())
            _error_check(err)
        else:
            object.__setattr__(self, name, value)

    def __getattr__(self, name):
        """Read text meta-data in the sound file through properties."""
        if name in _str_types:
            self._check_if_closed()
            data = _snd.sf_get_string(self._file, _str_types[name])
            return _ffi.string(data).decode('utf-8', 'replace') if data else ''
        else:
            raise AttributeError("'SoundFile' object has no attribute {0!r}".format(name))

    def __len__(self):
        return self._info.frames

    def __bool__(self):
        return True

    def __nonzero__(self):
        return self.__bool__()

    def seekable(self):
        """Return True if the file supports seeking."""
        return self._info.seekable == _snd.SF_TRUE

    def seek(self, frames, whence=SEEK_SET):
        """Set the read/write position.

        Parameters
        ----------
        frames : int
            The frame index or offset to seek.
        whence : {SEEK_SET, SEEK_CUR, SEEK_END}, optional
            By default (``whence=SEEK_SET``), *frames* are counted from
            the beginning of the file.
            ``whence=SEEK_CUR`` seeks from the current position
            (positive and negative values are allowed for *frames*).
            ``whence=SEEK_END`` seeks from the end (use negative value
            for *frames*).

        Returns
        -------
        int
            The new absolute read/write position in frames.

        Examples
        --------
        >>> from soundfile import SoundFile, SEEK_END
        >>> myfile = SoundFile('stereo_file.wav')

        Seek to the beginning of the file:

        >>> myfile.seek(0)
        0

        Seek to the end of the file:

        >>> myfile.seek(0, SEEK_END)
        44100  # this is the file length

        """
        self._check_if_closed()
        position = _snd.sf_seek(self._file, frames, whence)
        _error_check(self._errorcode)
        return position

    def tell(self):
        """Return the current read/write position."""
        return self.seek(0, SEEK_CUR)

    def read(self, frames=-1, dtype='float64', always_2d=False, fill_value=None, out=None):
        """Read from the file and return data as NumPy array.

        Reads the given number of frames in the given data format
        starting at the current read/write position.  This advances the
        read/write position by the same number of frames.
        By default, all frames from the current read/write position to
        the end of the file are returned.
        Use `seek()` to move the current read/write position.

        Parameters
        ----------
        frames : int, optional
            The number of frames to read. If ``frames < 0``, the whole
            rest of the file is read.
        dtype : {'float64', 'float32', 'int32', 'int16'}, optional
            Data type of the returned array, by default ``'float64'``.
            Floating point audio data is typically in the range from
            ``-1.0`` to ``1.0``. Integer data is in the range from
            ``-2**15`` to ``2**15-1`` for ``'int16'`` and from
            ``-2**31`` to ``2**31-1`` for ``'int32'``.

            .. note:: Reading int values from a float file will *not*
                scale the data to [-1.0, 1.0). If the file contains
                ``np.array([42.6], dtype='float32')``, you will read
                ``np.array([43], dtype='int32')`` for
                ``dtype='int32'``.

        Returns
        -------
        audiodata : `numpy.ndarray` or type(out)
            A two-dimensional NumPy (frames x channels) array is
            returned. If the sound file has only one channel, a
            one-dimensional array is returned. Use ``always_2d=True``
            to return a two-dimensional array anyway.

            If *out* was specified, it is returned. If *out* has more
            frames than available in the file (or if *frames* is
            smaller than the length of *out*) and no *fill_value* is
            given, then only a part of *out* is overwritten and a view
            containing all valid frames is returned.

        Other Parameters
        ----------------
        always_2d : bool, optional
            By default, reading a mono sound file will return a
            one-dimensional array. With ``always_2d=True``, audio data
            is always returned as a two-dimensional array, even if the
            audio file has only one channel.
        fill_value : float, optional
            If more frames are requested than available in the file,
            the rest of the output is be filled with *fill_value*. If
            *fill_value* is not specified, a smaller array is
            returned.
        out : `numpy.ndarray` or subclass, optional
            If *out* is specified, the data is written into the given
            array instead of creating a new array. In this case, the
            arguments *dtype* and *always_2d* are silently ignored! If
            *frames* is not given, it is obtained from the length of
            *out*.

        Examples
        --------
        >>> from soundfile import SoundFile
        >>> myfile = SoundFile('stereo_file.wav')

        Reading 3 frames from a stereo file:

        >>> myfile.read(3)
        array([[ 0.71329652,  0.06294799],
               [-0.26450912, -0.38874483],
               [ 0.67398441, -0.11516333]])
        >>> myfile.close()

        See Also
        --------
        buffer_read, .write

        """
        if out is None:
            frames = self._check_frames(frames, fill_value)
            out = self._create_empty_array(frames, always_2d, dtype)
        elif frames < 0 or frames > len(out):
            frames = len(out)
        frames = self._array_io('read', out, frames)
        if len(out) > frames:
            if fill_value is None:
                out = out[:frames]
            else:
                out[frames:] = fill_value
        return out

    def buffer_read(self, frames=-1, dtype=None):
        """Read from the file and return data as buffer object.

        Reads the given number of *frames* in the given data format
        starting at the current read/write position.  This advances the
        read/write position by the same number of frames.
        By default, all frames from the current read/write position to
        the end of the file are returned.
        Use `seek()` to move the current read/write position.

        Parameters
        ----------
        frames : int, optional
            The number of frames to read. If ``frames < 0``, the whole
            rest of the file is read.
        dtype : {'float64', 'float32', 'int32', 'int16'}
            Audio data will be converted to the given data type.

        Returns
        -------
        buffer
            A buffer containing the read data.

        See Also
        --------
        buffer_read_into, .read, buffer_write

        """
        frames = self._check_frames(frames, fill_value=None)
        ctype = self._check_dtype(dtype)
        cdata = _ffi.new(ctype + '[]', frames * self.channels)
        read_frames = self._cdata_io('read', cdata, ctype, frames)
        assert read_frames == frames
        return _ffi.buffer(cdata)

    def buffer_read_into(self, buffer, dtype):
        """Read from the file into a given buffer object.

        Fills the given *buffer* with frames in the given data format
        starting at the current read/write position (which can be
        changed with `seek()`) until the buffer is full or the end
        of the file is reached.  This advances the read/write position
        by the number of frames that were read.

        Parameters
        ----------
        buffer : writable buffer
            Audio frames from the file are written to this buffer.
        dtype : {'float64', 'float32', 'int32', 'int16'}
            The data type of *buffer*.

        Returns
        -------
        int
            The number of frames that were read from the file.
            This can be less than the size of *buffer*.
            The rest of the buffer is not filled with meaningful data.

        See Also
        --------
        buffer_read, .read

        """
        ctype = self._check_dtype(dtype)
        cdata, frames = self._check_buffer(buffer, ctype)
        frames = self._cdata_io('read', cdata, ctype, frames)
        return frames

    def write(self, data):
        """Write audio data from a NumPy array to the file.

        Writes a number of frames at the read/write position to the
        file. This also advances the read/write position by the same
        number of frames and enlarges the file if necessary.

        Note that writing int values to a float file will *not* scale
        the values to [-1.0, 1.0). If you write the value
        ``np.array([42], dtype='int32')``, to a ``subtype='FLOAT'``
        file, the file will then contain ``np.array([42.],
        dtype='float32')``.

        Parameters
        ----------
        data : array_like
            The data to write. Usually two-dimensional (frames x
            channels), but one-dimensional *data* can be used for mono
            files. Only the data types ``'float64'``, ``'float32'``,
            ``'int32'`` and ``'int16'`` are supported.

            .. note:: The data type of *data* does **not** select the
                  data type of the written file. Audio data will be
                  converted to the given *subtype*. Writing int values
                  to a float file will *not* scale the values to
                  [-1.0, 1.0). If you write the value ``np.array([42],
                  dtype='int32')``, to a ``subtype='FLOAT'`` file, the
                  file will then contain ``np.array([42.],
                  dtype='float32')``.

        Examples
        --------
        >>> import numpy as np
        >>> from soundfile import SoundFile
        >>> myfile = SoundFile('stereo_file.wav')

        Write 10 frames of random data to a new file:

        >>> with SoundFile('stereo_file.wav', 'w', 44100, 2, 'PCM_24') as f:
        >>>     f.write(np.random.randn(10, 2))

        See Also
        --------
        buffer_write, .read

        """
        import numpy as np
        data = np.ascontiguousarray(data)
        written = self._array_io('write', data, len(data))
        assert written == len(data)
        self._update_frames(written)

    def buffer_write(self, data, dtype):
        """Write audio data from a buffer/bytes object to the file.

        Writes the contents of *data* to the file at the current
        read/write position.
        This also advances the read/write position by the number of
        frames that were written and enlarges the file if necessary.

        Parameters
        ----------
        data : buffer or bytes
            A buffer or bytes object containing the audio data to be
            written.
        dtype : {'float64', 'float32', 'int32', 'int16'}
            The data type of the audio data stored in *data*.

        See Also
        --------
        .write, buffer_read

        """
        ctype = self._check_dtype(dtype)
        cdata, frames = self._check_buffer(data, ctype)
        written = self._cdata_io('write', cdata, ctype, frames)
        assert written == frames
        self._update_frames(written)

    def blocks(self, blocksize=None, overlap=0, frames=-1, dtype='float64', always_2d=False, fill_value=None, out=None):
        """Return a generator for block-wise reading.

        By default, the generator yields blocks of the given
        *blocksize* (using a given *overlap*) until the end of the file
        is reached; *frames* can be used to stop earlier.

        Parameters
        ----------
        blocksize : int
            The number of frames to read per block. Either this or *out*
            must be given.
        overlap : int, optional
            The number of frames to rewind between each block.
        frames : int, optional
            The number of frames to read.
            If ``frames < 0``, the file is read until the end.
        dtype : {'float64', 'float32', 'int32', 'int16'}, optional
            See `read()`.

        Yields
        ------
        `numpy.ndarray` or type(out)
            Blocks of audio data.
            If *out* was given, and the requested frames are not an
            integer multiple of the length of *out*, and no
            *fill_value* was given, the last block will be a smaller
            view into *out*.


        Other Parameters
        ----------------
        always_2d, fill_value, out
            See `read()`.
        fill_value : float, optional
            See `read()`.
        out : `numpy.ndarray` or subclass, optional
            If *out* is specified, the data is written into the given
            array instead of creating a new array. In this case, the
            arguments *dtype* and *always_2d* are silently ignored!

        Examples
        --------
        >>> from soundfile import SoundFile
        >>> with SoundFile('stereo_file.wav') as f:
        >>>     for block in f.blocks(blocksize=1024):
        >>>         pass  # do something with 'block'

        """
        import numpy as np
        if 'r' not in self.mode and '+' not in self.mode:
            raise SoundFileRuntimeError('blocks() is not allowed in write-only mode')
        if out is None:
            if blocksize is None:
                raise TypeError('One of {blocksize, out} must be specified')
            out = self._create_empty_array(blocksize, always_2d, dtype)
            copy_out = True
        else:
            if blocksize is not None:
                raise TypeError('Only one of {blocksize, out} may be specified')
            blocksize = len(out)
            copy_out = False
        overlap_memory = None
        frames = self._check_frames(frames, fill_value)
        while frames > 0:
            if overlap_memory is None:
                output_offset = 0
            else:
                output_offset = len(overlap_memory)
                out[:output_offset] = overlap_memory
            toread = min(blocksize - output_offset, frames)
            self.read(toread, dtype, always_2d, fill_value, out[output_offset:])
            if overlap:
                if overlap_memory is None:
                    overlap_memory = np.copy(out[-overlap:])
                else:
                    overlap_memory[:] = out[-overlap:]
            if blocksize > frames + overlap and fill_value is None:
                block = out[:frames + overlap]
            else:
                block = out
            yield (np.copy(block) if copy_out else block)
            frames -= toread

    def truncate(self, frames=None):
        """Truncate the file to a given number of frames.

        After this command, the read/write position will be at the new
        end of the file.

        Parameters
        ----------
        frames : int, optional
            Only the data before *frames* is kept, the rest is deleted.
            If not specified, the current read/write position is used.

        """
        if frames is None:
            frames = self.tell()
        err = _snd.sf_command(self._file, _snd.SFC_FILE_TRUNCATE, _ffi.new('sf_count_t*', frames), _ffi.sizeof('sf_count_t'))
        if err:
            err = _snd.sf_error(self._file)
            raise LibsndfileError(err, 'Error truncating the file')
        self._info.frames = frames

    def flush(self):
        """Write unwritten data to the file system.

        Data written with `write()` is not immediately written to
        the file system but buffered in memory to be written at a later
        time.  Calling `flush()` makes sure that all changes are
        actually written to the file system.

        This has no effect on files opened in read-only mode.

        """
        self._check_if_closed()
        _snd.sf_write_sync(self._file)

    def close(self):
        """Close the file.  Can be called multiple times."""
        if not self.closed:
            self.flush()
            err = _snd.sf_close(self._file)
            self._file = None
            _error_check(err)

    def _open(self, file, mode_int, closefd):
        """Call the appropriate sf_open*() function from libsndfile."""
        if isinstance(file, (_unicode, bytes)):
            if _os.path.isfile(file):
                if 'x' in self.mode:
                    raise OSError('File exists: {0!r}'.format(self.name))
                elif set(self.mode).issuperset('w+'):
                    _os.close(_os.open(file, _os.O_WRONLY | _os.O_TRUNC))
            openfunction = _snd.sf_open
            if isinstance(file, _unicode):
                if _sys.platform == 'win32':
                    openfunction = _snd.sf_wchar_open
                else:
                    file = file.encode(_sys.getfilesystemencoding())
            file_ptr = openfunction(file, mode_int, self._info)
        elif isinstance(file, int):
            file_ptr = _snd.sf_open_fd(file, mode_int, self._info, closefd)
        elif _has_virtual_io_attrs(file, mode_int):
            file_ptr = _snd.sf_open_virtual(self._init_virtual_io(file), mode_int, self._info, _ffi.NULL)
        else:
            raise TypeError('Invalid file: {0!r}'.format(self.name))
        if file_ptr == _ffi.NULL:
            err = _snd.sf_error(file_ptr)
            raise LibsndfileError(err, prefix='Error opening {0!r}: '.format(self.name))
        if mode_int == _snd.SFM_WRITE:
            self._info.frames = 0
        return file_ptr

    def _init_virtual_io(self, file):
        """Initialize callback functions for sf_open_virtual()."""

        @_ffi.callback('sf_vio_get_filelen')
        def vio_get_filelen(user_data):
            curr = file.tell()
            file.seek(0, SEEK_END)
            size = file.tell()
            file.seek(curr, SEEK_SET)
            return size

        @_ffi.callback('sf_vio_seek')
        def vio_seek(offset, whence, user_data):
            file.seek(offset, whence)
            return file.tell()

        @_ffi.callback('sf_vio_read')
        def vio_read(ptr, count, user_data):
            try:
                buf = _ffi.buffer(ptr, count)
                data_read = file.readinto(buf)
            except AttributeError:
                data = file.read(count)
                data_read = len(data)
                buf = _ffi.buffer(ptr, data_read)
                buf[0:data_read] = data
            return data_read

        @_ffi.callback('sf_vio_write')
        def vio_write(ptr, count, user_data):
            buf = _ffi.buffer(ptr, count)
            data = buf[:]
            written = file.write(data)
            if written is None:
                written = count
            return written

        @_ffi.callback('sf_vio_tell')
        def vio_tell(user_data):
            return file.tell()
        self._virtual_io = {'get_filelen': vio_get_filelen, 'seek': vio_seek, 'read': vio_read, 'write': vio_write, 'tell': vio_tell}
        return _ffi.new('SF_VIRTUAL_IO*', self._virtual_io)

    def _getAttributeNames(self):
        """Return all attributes used in __setattr__ and __getattr__.

        This is useful for auto-completion (e.g. IPython).

        """
        return _str_types

    def _check_if_closed(self):
        """Check if the file is closed and raise an error if it is.

        This should be used in every method that uses self._file.

        """
        if self.closed:
            raise SoundFileRuntimeError('I/O operation on closed file')

    def _check_frames(self, frames, fill_value):
        """Reduce frames to no more than are available in the file."""
        if self.seekable():
            remaining_frames = self.frames - self.tell()
            if frames < 0 or (frames > remaining_frames and fill_value is None):
                frames = remaining_frames
        elif frames < 0:
            raise ValueError('frames must be specified for non-seekable files')
        return frames

    def _check_buffer(self, data, ctype):
        """Convert buffer to cdata and check for valid size."""
        assert ctype in _ffi_types.values()
        if not isinstance(data, bytes):
            data = _ffi.from_buffer(data)
        frames, remainder = divmod(len(data), self.channels * _ffi.sizeof(ctype))
        if remainder:
            raise ValueError('Data size must be a multiple of frame size')
        return (data, frames)

    def _create_empty_array(self, frames, always_2d, dtype):
        """Create an empty array with appropriate shape."""
        import numpy as np
        if always_2d or self.channels > 1:
            shape = (frames, self.channels)
        else:
            shape = (frames,)
        return np.empty(shape, dtype, order='C')

    def _check_dtype(self, dtype):
        """Check if dtype string is valid and return ctype string."""
        try:
            return _ffi_types[dtype]
        except KeyError:
            raise ValueError('dtype must be one of {0!r} and not {1!r}'.format(sorted(_ffi_types.keys()), dtype))

    def _array_io(self, action, array, frames):
        """Check array and call low-level IO function."""
        if array.ndim not in (1, 2) or (array.ndim == 1 and self.channels != 1) or (array.ndim == 2 and array.shape[1] != self.channels):
            raise ValueError('Invalid shape: {0!r}'.format(array.shape))
        if not array.flags.c_contiguous:
            raise ValueError('Data must be C-contiguous')
        ctype = self._check_dtype(array.dtype.name)
        assert array.dtype.itemsize == _ffi.sizeof(ctype)
        cdata = _ffi.cast(ctype + '*', array.__array_interface__['data'][0])
        return self._cdata_io(action, cdata, ctype, frames)

    def _cdata_io(self, action, data, ctype, frames):
        """Call one of libsndfile's read/write functions."""
        assert ctype in _ffi_types.values()
        self._check_if_closed()
        if self.seekable():
            curr = self.tell()
        func = getattr(_snd, 'sf_' + action + 'f_' + ctype)
        frames = func(self._file, data, frames)
        _error_check(self._errorcode)
        if self.seekable():
            self.seek(curr + frames, SEEK_SET)
        return frames

    def _update_frames(self, written):
        """Update self.frames after writing."""
        if self.seekable():
            curr = self.tell()
            self._info.frames = self.seek(0, SEEK_END)
            self.seek(curr, SEEK_SET)
        else:
            self._info.frames += written

    def _prepare_read(self, start, stop, frames):
        """Seek to start frame and calculate length."""
        if start != 0 and (not self.seekable()):
            raise ValueError('start is only allowed for seekable files')
        if frames >= 0 and stop is not None:
            raise TypeError('Only one of {frames, stop} may be used')
        start, stop, _ = slice(start, stop).indices(self.frames)
        if stop < start:
            stop = start
        if frames < 0:
            frames = stop - start
        if self.seekable():
            self.seek(start, SEEK_SET)
        return frames

    def copy_metadata(self):
        """Get all metadata present in this SoundFile

        Returns
        -------

        metadata: dict[str, str]
            A dict with all metadata. Possible keys are: 'title', 'copyright',
            'software', 'artist', 'comment', 'date', 'album', 'license',
            'tracknumber' and 'genre'.
        """
        strs = {}
        for strtype, strid in _str_types.items():
            data = _snd.sf_get_string(self._file, strid)
            if data:
                strs[strtype] = _ffi.string(data).decode('utf-8', 'replace')
        return strs