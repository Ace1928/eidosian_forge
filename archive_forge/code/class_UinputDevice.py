import libevdev
import os
import ctypes
import errno
from ctypes import c_char_p
from ctypes import c_int
from ctypes import c_uint
from ctypes import c_void_p
from ctypes import c_long
from ctypes import c_int32
from ctypes import c_uint16
class UinputDevice(_LibraryWrapper):
    """
    This class provides a wrapper around the libevdev C library

    .. warning::

        Do not use this class directly
    """

    @staticmethod
    def _cdll():
        return ctypes.CDLL('libevdev.so.2', use_errno=True)
    _api_prototypes = {'libevdev_uinput_create_from_device': {'argtypes': (c_void_p, c_int, ctypes.POINTER(ctypes.POINTER(_UinputDevice))), 'restype': c_int}, 'libevdev_uinput_destroy': {'argtypes': (c_void_p,), 'restype': None}, 'libevdev_uinput_get_devnode': {'argtypes': (c_void_p,), 'restype': c_char_p}, 'libevdev_uinput_get_syspath': {'argtypes': (c_void_p,), 'restype': c_char_p}, 'libevdev_uinput_write_event': {'argtypes': (c_void_p, c_uint, c_uint, c_int), 'restype': c_int}}

    def __init__(self, source, fileobj=None):
        """
        Create a new uinput device based on the source libevdev device. The
        uinput device will mirror all capabilities from the source device.

        :param source: A libevdev device with all capabilities set.
        :param fileobj: A file-like object to the /dev/uinput node. If None,
        libevdev will open the device in managed mode. See the libevdev
        documentation for details.
        """
        super(UinputDevice, self).__init__()
        self._fileobj = fileobj
        if fileobj is None:
            fd = -2
        else:
            fd = fileobj.fileno()
        self._uinput_device = ctypes.POINTER(_UinputDevice)()
        rc = self._uinput_create_from_device(source._ctx, fd, ctypes.byref(self._uinput_device))
        if rc != 0:
            raise OSError(-rc, os.strerror(-rc))

    def __del__(self):
        if self._uinput_device is not None:
            self._uinput_destroy(self._uinput_device)

    @property
    def fd(self):
        """
        :return: the file-like object used in the constructor.
        """
        return self.fileobj

    def write_event(self, type, code, value):
        self._uinput_write_event(self._uinput_device, type, code, value)

    @property
    def devnode(self):
        """
        Return a string with the /dev/input/eventX device node
        """
        devnode = self._uinput_get_devnode(self._uinput_device)
        return devnode.decode('iso8859-1')

    @property
    def syspath(self):
        """
        Return a string with the /dev/input/eventX device node
        """
        syspath = self._uinput_get_syspath(self._uinput_device)
        return syspath.decode('iso8859-1')