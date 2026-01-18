import os
import sys
import ctypes
import threading
import logging
import numpy
from ..core import (
class Freeimage(object):
    """Class to represent an interface to the FreeImage library.
    This class is relatively thin. It provides a Pythonic API that converts
    Freeimage objects to Python objects, but that's about it.
    The actual implementation should be provided by the plugins.

    The recommended way to call into the Freeimage library (so that
    errors and warnings show up in the right moment) is to use this
    object as a context manager:
    with imageio.fi as lib:
        lib.FreeImage_GetPalette()

    """
    _API = {'FreeImage_AllocateT': (ctypes.c_void_p, None), 'FreeImage_FindFirstMetadata': (ctypes.c_void_p, None), 'FreeImage_GetBits': (ctypes.c_void_p, None), 'FreeImage_GetPalette': (ctypes.c_void_p, None), 'FreeImage_GetTagKey': (ctypes.c_char_p, None), 'FreeImage_GetTagValue': (ctypes.c_void_p, None), 'FreeImage_CreateTag': (ctypes.c_void_p, None), 'FreeImage_Save': (ctypes.c_void_p, None), 'FreeImage_Load': (ctypes.c_void_p, None), 'FreeImage_LoadFromMemory': (ctypes.c_void_p, None), 'FreeImage_OpenMultiBitmap': (ctypes.c_void_p, None), 'FreeImage_LoadMultiBitmapFromMemory': (ctypes.c_void_p, None), 'FreeImage_LockPage': (ctypes.c_void_p, None), 'FreeImage_OpenMemory': (ctypes.c_void_p, None), 'FreeImage_GetVersion': (ctypes.c_char_p, None), 'FreeImage_GetFIFExtensionList': (ctypes.c_char_p, None), 'FreeImage_GetFormatFromFIF': (ctypes.c_char_p, None), 'FreeImage_GetFIFDescription': (ctypes.c_char_p, None), 'FreeImage_ColorQuantizeEx': (ctypes.c_void_p, None), 'FreeImage_IsLittleEndian': (ctypes.c_int, None), 'FreeImage_SetOutputMessage': (ctypes.c_void_p, None), 'FreeImage_GetFIFCount': (ctypes.c_int, None), 'FreeImage_IsPluginEnabled': (ctypes.c_int, None), 'FreeImage_GetFileType': (ctypes.c_int, None), 'FreeImage_GetTagType': (ctypes.c_int, None), 'FreeImage_GetTagLength': (ctypes.c_int, None), 'FreeImage_FindNextMetadata': (ctypes.c_int, None), 'FreeImage_FindCloseMetadata': (ctypes.c_void_p, None), 'FreeImage_GetFIFFromFilename': (ctypes.c_int, None), 'FreeImage_FIFSupportsReading': (ctypes.c_int, None), 'FreeImage_FIFSupportsWriting': (ctypes.c_int, None), 'FreeImage_FIFSupportsExportType': (ctypes.c_int, None), 'FreeImage_FIFSupportsExportBPP': (ctypes.c_int, None), 'FreeImage_GetHeight': (ctypes.c_int, None), 'FreeImage_GetWidth': (ctypes.c_int, None), 'FreeImage_GetImageType': (ctypes.c_int, None), 'FreeImage_GetBPP': (ctypes.c_int, None), 'FreeImage_GetColorsUsed': (ctypes.c_int, None), 'FreeImage_ConvertTo32Bits': (ctypes.c_void_p, None), 'FreeImage_GetPitch': (ctypes.c_int, None), 'FreeImage_Unload': (ctypes.c_void_p, None)}

    def __init__(self):
        self._lib = None
        self._lock = threading.RLock()
        self._messages = []
        if sys.platform.startswith('win'):
            functype = ctypes.WINFUNCTYPE
        else:
            functype = ctypes.CFUNCTYPE

        @functype(None, ctypes.c_int, ctypes.c_char_p)
        def error_handler(fif, message):
            message = message.decode('utf-8')
            self._messages.append(message)
            while len(self._messages) > 256:
                self._messages.pop(0)
        self._error_handler = error_handler

    @property
    def lib(self):
        if self._lib is None:
            try:
                self.load_freeimage()
            except OSError as err:
                self._lib = 'The freeimage library could not be loaded: '
                self._lib += str(err)
        if isinstance(self._lib, str):
            raise RuntimeError(self._lib)
        return self._lib

    def has_lib(self):
        try:
            self.lib
        except Exception:
            return False
        return True

    def load_freeimage(self):
        """Try to load the freeimage lib from the system. If not successful,
        try to download the imageio version and try again.
        """
        success = False
        try:
            self._load_freeimage()
            self._register_api()
            if self.lib.FreeImage_GetVersion().decode('utf-8') >= '3.15':
                success = True
        except OSError:
            pass
        if not success:
            get_freeimage_lib()
            self._load_freeimage()
            self._register_api()
        self.lib.FreeImage_SetOutputMessage(self._error_handler)
        self.lib_version = self.lib.FreeImage_GetVersion().decode('utf-8')

    def _load_freeimage(self):
        lib_names = ['freeimage', 'libfreeimage']
        exact_lib_names = ['FreeImage', 'libfreeimage.dylib', 'libfreeimage.so', 'libfreeimage.so.3']
        res_dirs = resource_dirs()
        plat = get_platform()
        if plat:
            fname = FNAME_PER_PLATFORM[plat]
            for dir in res_dirs:
                exact_lib_names.insert(0, os.path.join(dir, 'freeimage', fname))
        lib = os.getenv('IMAGEIO_FREEIMAGE_LIB', None)
        if lib is not None:
            exact_lib_names.insert(0, lib)
        try:
            lib, fname = load_lib(exact_lib_names, lib_names, res_dirs)
        except OSError as err:
            err_msg = str(err) + '\nPlease install the FreeImage library.'
            raise OSError(err_msg)
        self._lib = lib
        self.lib_fname = fname

    def _register_api(self):
        for f, (restype, argtypes) in self._API.items():
            func = getattr(self.lib, f)
            func.restype = restype
            func.argtypes = argtypes

    def __enter__(self):
        self._lock.acquire()
        return self.lib

    def __exit__(self, *args):
        self._show_any_warnings()
        self._lock.release()

    def _reset_log(self):
        """Reset the list of output messages. Call this before
        loading or saving an image with the FreeImage API.
        """
        self._messages = []

    def _get_error_message(self):
        """Get the output messages produced since the last reset as
        one string. Returns 'No known reason.' if there are no messages.
        Also resets the log.
        """
        if self._messages:
            res = ' '.join(self._messages)
            self._reset_log()
            return res
        else:
            return 'No known reason.'

    def _show_any_warnings(self):
        """If there were any messages since the last reset, show them
        as a warning. Otherwise do nothing. Also resets the messages.
        """
        if self._messages:
            logger.warning('imageio.freeimage warning: ' + self._get_error_message())
            self._reset_log()

    def get_output_log(self):
        """Return a list of the last 256 output messages
        (warnings and errors) produced by the FreeImage library.
        """
        return [m for m in self._messages]

    def getFIF(self, filename, mode, bb=None):
        """Get the freeimage Format (FIF) from a given filename.
        If mode is 'r', will try to determine the format by reading
        the file, otherwise only the filename is used.

        This function also tests whether the format supports reading/writing.
        """
        with self as lib:
            ftype = -1
            if mode not in 'rw':
                raise ValueError('Invalid mode (must be "r" or "w").')
            if mode == 'r':
                if bb is not None:
                    fimemory = lib.FreeImage_OpenMemory(ctypes.c_char_p(bb), len(bb))
                    ftype = lib.FreeImage_GetFileTypeFromMemory(ctypes.c_void_p(fimemory), len(bb))
                    lib.FreeImage_CloseMemory(ctypes.c_void_p(fimemory))
                if ftype == -1 and os.path.isfile(filename):
                    ftype = lib.FreeImage_GetFileType(efn(filename), 0)
            if ftype == -1:
                ftype = lib.FreeImage_GetFIFFromFilename(efn(filename))
            if ftype == -1:
                raise ValueError('Cannot determine format of file "%s"' % filename)
            elif mode == 'w' and (not lib.FreeImage_FIFSupportsWriting(ftype)):
                raise ValueError('Cannot write the format of file "%s"' % filename)
            elif mode == 'r' and (not lib.FreeImage_FIFSupportsReading(ftype)):
                raise ValueError('Cannot read the format of file "%s"' % filename)
            return ftype

    def create_bitmap(self, filename, ftype, flags=0):
        """create_bitmap(filename, ftype, flags=0)
        Create a wrapped bitmap object.
        """
        return FIBitmap(self, filename, ftype, flags)

    def create_multipage_bitmap(self, filename, ftype, flags=0):
        """create_multipage_bitmap(filename, ftype, flags=0)
        Create a wrapped multipage bitmap object.
        """
        return FIMultipageBitmap(self, filename, ftype, flags)