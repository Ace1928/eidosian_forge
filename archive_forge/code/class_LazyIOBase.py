from ._base import *
class LazyIOBase(object):
    __metaclass__ = ABCMeta

    def __init__(self, filename: str, mode: ModeIO=ModeIO.auto, allow_deletion: bool=False, *args, **kwargs):
        self._setup_file(filename)
        self._mode = ModeIO.append if mode == 'auto' and self.exists else mode
        self._io = None
        self._io_closed = True
        self._allow_rm = allow_deletion
        self._args, self._kwargs = (args, kwargs)
        self.open()

    def _setup_file(self, filename: str=None):
        if filename is None:
            return
        self._filename = filename
        self._basename = File.base(filename)
        self._directory = File.getdir(filename)
        File.mkdirs(self._directory)
        self._fext = File.ext(self._filename)
        self._ext = ExtIO(self._fext)

    def open(self, filename: str=None, mode: ModeIO=None):
        mode = mode.value or self.mode
        filename = filename or self._filename
        if not self.is_closed:
            self.close()
        self._io = gfile(self._filename, mode)
        self._io_closed = False
        if not self.exists:
            self._io.write()
            self.flush()

    def close(self):
        if self._io is None:
            return
        self._io.flush()
        self._io.close()
        self._io_closed = True
        self._io = None

    def flush(self):
        self._io.flush()

    def _write(self, data, *args, **kwargs):
        self._io.write(data)

    def write(self, data, *args, **kwargs):
        self._ensure_open()
        self._write(data, *args, **kwargs)

    def _read(self, *args, **kwargs):
        return self._io.read(*args, **kwargs)

    def read(self, *args, **kwargs):
        self._ensure_open()
        return self._read(*args, **kwargs)

    def _readlines(self, *args, **kwargs):
        return self._io.readlines()

    def readlines(self, *args, **kwargs):
        self._ensure_open()
        return self._readlines(*args, **kwargs)

    @timed_cache(120)
    def get_num_lines(self):
        return sum((1 for _ in File.tflines(self._filename)))

    @property
    def filesize(self):
        self._ensure_open()
        return self._io.size()

    @timed_cache(10)
    def seek(self, offset=None, whence=0, position=None):
        self._ensure_open()
        return self._io.seek(offset, whence, position)

    def _readline(self, *args, **kwargs):
        return self._io.readline()

    def readline(self, *args, **kwargs):
        self._ensure_open()
        return self._readline(*args, **kwargs)

    def _readlines(self, *args, **kwargs):
        return self._io.readline()

    def readlines(self, *args, **kwargs):
        self._ensure_open()
        return self._readlines(*args, **kwargs)

    def _tell(self, *args, **kwargs):
        return self._io.tell()

    def tell(self, *args, **kwargs):
        self._ensure_open()
        return self._tell(*args, **kwargs)

    def _iterator(self):
        return self

    def __iter__(self):
        return self._iterator()

    def _getnext(self):
        return self._io.next()

    def __next__(self):
        return self._getnext()

    @property
    def seekable(self):
        self._ensure_open()
        return self._io.seekable()

    def setmode(self, mode: ModeIO=ModeIO.auto):
        if mode.value != self._mode:
            self.close()
        self._mode = mode.value
        self.open()

    def setfile(self, filename: str, mode: ModeIO=None, allow_deletion: bool=False, *args, **kwargs):
        self.close()
        self._setup_file(filename)
        if mode is not None:
            self.setmode(mode)
        self._allow_rm = self._allow_rm or allow_deletion
        self._args = args
        if kwargs:
            self._kwargs.update(kwargs)
        self.open()

    @classmethod
    def modify(cls, filename, new_filename=None, prefix=None, suffix=None, extension=None, directory=None, create_dirs=True, filename_only=False, space_replace=None):
        return File.mod_fname(filename=filename, newname=new_filename, prefix=prefix, suffix=suffix, ext=extension, directory=directory, create_dirs=create_dirs, filename_only=filename_only, space_replace=space_replace)

    @timed_cache(15)
    def _ensure_open(self):
        if self.is_closed:
            raise ValueError('File is closed.')

    def compare(self, target_filename):
        from tensorflow.python.lib.io.file_io import filecmp
        return filecmp(self._filename, target_filename)

    @property
    def exists(self):
        return File.exists(self._filename)

    @property
    def mode(self):
        return self._mode.value

    @property
    def is_closed(self):
        return self._io_closed

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        self.backend = None

    def __del__(self):
        self.close()
        if not self._allow_rm:
            raise ValueError('Config allow_deletion = True must be set to allow del')
        File.rm(self._filename)

    def backup(self, filepath: str=None, directory: str=None, suffix: str='timestamp', overwrite: bool=False):
        if suffix == 'timestamp':
            suffix = tstamp()
        backup_fname = self.modify(self._filename, filepath=filepath, directory=directory, suffix=suffix)
        try:
            File.copy(self._filename, backup_fname, overwrite=overwrite)
        except Exception as e:
            logger.error(str(e))
            return None
        return backup_fname