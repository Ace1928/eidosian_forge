import os
import re
from copy import deepcopy
import numpy as np
from .arrayproxy import ArrayProxy
from .fileslice import strided_scalar
from .spatialimages import HeaderDataError, ImageDataError, SpatialHeader, SpatialImage
from .volumeutils import Recoder
class AFNIImage(SpatialImage):
    """
    AFNI Image file

    Can be loaded from either the BRIK or HEAD file (but MUST specify one!)

    Examples
    --------
    >>> import nibabel as nib
    >>> brik = nib.load(os.path.join(datadir, 'example4d+orig.BRIK.gz'))
    >>> brik.shape
    (33, 41, 25, 3)
    >>> brik.affine
    array([[ -3.    ,  -0.    ,  -0.    ,  49.5   ],
           [ -0.    ,  -3.    ,  -0.    ,  82.312 ],
           [  0.    ,   0.    ,   3.    , -52.3511],
           [  0.    ,   0.    ,   0.    ,   1.    ]])
    >>> head = load(os.path.join(datadir, 'example4d+orig.HEAD'))
    >>> np.array_equal(head.get_fdata(), brik.get_fdata())
    True
    """
    header_class = AFNIHeader
    header: AFNIHeader
    valid_exts = ('.brik', '.head')
    files_types = (('image', '.brik'), ('header', '.head'))
    _compressed_suffixes = ('.gz', '.bz2', '.Z', '.zst')
    makeable = False
    rw = False
    ImageArrayProxy = AFNIArrayProxy

    @classmethod
    def from_file_map(klass, file_map, *, mmap=True, keep_file_open=None):
        """
        Creates an AFNIImage instance from `file_map`

        Parameters
        ----------
        file_map : dict
            dict with keys ``image, header`` and values being fileholder
            objects for the respective BRIK and HEAD files
        mmap : {True, False, 'c', 'r'}, optional, keyword only
            `mmap` controls the use of numpy memory mapping for reading image
            array data.  If False, do not try numpy ``memmap`` for data array.
            If one of {'c', 'r'}, try numpy memmap with ``mode=mmap``.  A
            `mmap` value of True gives the same behavior as ``mmap='c'``.  If
            image data file cannot be memory-mapped, ignore `mmap` value and
            read array from file.
        keep_file_open : {None, True, False}, optional, keyword only
            `keep_file_open` controls whether a new file handle is created
            every time the image is accessed, or a single file handle is
            created and used for the lifetime of this ``ArrayProxy``. If
            ``True``, a single file handle is created and used. If ``False``,
            a new file handle is created every time the image is accessed.
            If ``file_like`` refers to an open file handle, this setting has no
            effect. The default value (``None``) will result in the value of
            ``nibabel.arrayproxy.KEEP_FILE_OPEN_DEFAULT`` being used.
        """
        with file_map['header'].get_prepare_fileobj('rt') as hdr_fobj:
            hdr = klass.header_class.from_fileobj(hdr_fobj)
        imgf = file_map['image'].fileobj
        imgf = file_map['image'].filename if imgf is None else imgf
        data = klass.ImageArrayProxy(imgf, hdr.copy(), mmap=mmap, keep_file_open=keep_file_open)
        return klass(data, hdr.get_affine(), header=hdr, extra=None, file_map=file_map)

    @classmethod
    def filespec_to_file_map(klass, filespec):
        """
        Make `file_map` from filename `filespec`

        AFNI BRIK files can be compressed, but HEAD files cannot - see
        afni.nimh.nih.gov/pub/dist/doc/program_help/README.compression.html.
        Thus, if you have AFNI files my_image.HEAD and my_image.BRIK.gz and you
        want to load the AFNI BRIK / HEAD pair, you can specify:

            * The HEAD filename - e.g., my_image.HEAD
            * The BRIK filename w/o compressed extension - e.g., my_image.BRIK
            * The full BRIK filename - e.g., my_image.BRIK.gz

        Parameters
        ----------
        filespec : str
            Filename that might be for this image file type.

        Returns
        -------
        file_map : dict
            dict with keys ``image`` and ``header`` where values are fileholder
            objects for the respective BRIK and HEAD files

        Raises
        ------
        ImageFileError
            If `filespec` is not recognizable as being a filename for this
            image type.
        """
        file_map = super().filespec_to_file_map(filespec)
        for key, fholder in file_map.items():
            fname = fholder.filename
            if key == 'header' and (not os.path.exists(fname)):
                for ext in klass._compressed_suffixes:
                    fname = fname[:-len(ext)] if fname.endswith(ext) else fname
            elif key == 'image' and (not os.path.exists(fname)):
                for ext in klass._compressed_suffixes:
                    if os.path.exists(fname + ext):
                        fname += ext
                        break
            file_map[key].filename = fname
        return file_map