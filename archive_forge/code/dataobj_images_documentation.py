from __future__ import annotations
import typing as ty
import numpy as np
from .arrayproxy import ArrayLike
from .deprecated import deprecate_with_version
from .filebasedimages import FileBasedHeader, FileBasedImage
from .fileholders import FileMap
Class method to create image from filename `filename`

        Parameters
        ----------
        filename : str
            Filename of image to load
        mmap : {True, False, 'c', 'r'}, optional, keyword only
            `mmap` controls the use of numpy memory mapping for reading image
            array data.  If False, do not try numpy ``memmap`` for data array.
            If one of {'c', 'r'}, try numpy memmap with ``mode=mmap``.  A
            `mmap` value of True gives the same behavior as ``mmap='c'``.  If
            image data file cannot be memory-mapped, ignore `mmap` value and
            read array from file.
        keep_file_open : { None, True, False }, optional, keyword only
            `keep_file_open` controls whether a new file handle is created
            every time the image is accessed, or a single file handle is
            created and used for the lifetime of this ``ArrayProxy``. If
            ``True``, a single file handle is created and used. If ``False``,
            a new file handle is created every time the image is accessed.
            The default value (``None``) will result in the value of
            ``nibabel.arrayproxy.KEEP_FILE_OPEN_DEFAULT`` being used.

        Returns
        -------
        img : DataobjImage instance
        