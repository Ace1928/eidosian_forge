from __future__ import annotations
import io
import typing as ty
from copy import deepcopy
from urllib import request
from ._compression import COMPRESSION_ERRORS
from .fileholders import FileHolder, FileMap
from .filename_parser import TypesFilenamesError, _stringify_path, splitext_addext, types_filenames
from .openers import ImageOpener
@classmethod
def _sniff_meta_for(klass, filename: FileSpec, sniff_nbytes: int, sniff: FileSniff | None=None) -> FileSniff | None:
    """Sniff metadata for image represented by `filename`

        Parameters
        ----------
        filename : str or os.PathLike
            Filename for an image, or an image header (metadata) file.
            If `filename` points to an image data file, and the image type has
            a separate "header" file, we work out the name of the header file,
            and read from that instead of `filename`.
        sniff_nbytes : int
            Number of bytes to read from the image or metadata file
        sniff : (bytes, fname), optional
            The result of a previous call to `_sniff_meta_for`.  If fname
            matches the computed header file name, `sniff` is returned without
            rereading the file.

        Returns
        -------
        sniff : None or (bytes, fname)
            None if we could not read the image or metadata file.  `sniff[0]`
            is either length `sniff_nbytes` or the length of the image /
            metadata file, whichever is the shorter. `fname` is the name of
            the sniffed file.
        """
    froot, ext, trailing = splitext_addext(filename, klass._compressed_suffixes)
    t_fnames = types_filenames(filename, klass.files_types, trailing_suffixes=klass._compressed_suffixes)
    meta_fname = t_fnames.get('header', _stringify_path(filename))
    if sniff is not None and sniff[1] == meta_fname:
        return sniff
    try:
        with ImageOpener(meta_fname, 'rb') as fobj:
            binaryblock = fobj.read(sniff_nbytes)
    except COMPRESSION_ERRORS + (OSError, EOFError):
        return None
    return (binaryblock, meta_fname)