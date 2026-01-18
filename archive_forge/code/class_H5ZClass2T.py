from ctypes import (
import pytest
import h5py
from h5py import h5z
from .common import insubprocess
class H5ZClass2T(Structure):
    """H5Z_class2_t structure defining a filter"""
    _fields_ = [('version', c_int), ('id_', c_int), ('encoder_present', c_uint), ('decoder_present', c_uint), ('name', c_char_p), ('can_apply', c_void_p), ('set_local', c_void_p), ('filter_', H5ZFuncT)]