import enum
import os
import platform
import sys
import cffi
def _mask_to_caps(mask):
    """Convert bitmask to list of set bit offsets"""
    return [i for i in range(64) if 1 << i & mask]