import contextlib
import ctypes
from ctypes import (
from ctypes.util import find_library
class error:
    item_not_found = -25300
    keychain_denied = -128
    sec_auth_failed = -25293
    plist_missing = -67030
    sec_interaction_not_allowed = -25308