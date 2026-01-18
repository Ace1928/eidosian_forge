import errno
import locale
import logging
import os
import stat
import sys
import time
from optparse import Option, OptionParser
import nibabel as nib
import nibabel.dft as dft
class dummy_fuse:
    """Dummy fuse "module" so that nose does not blow during doctests"""
    Fuse = object