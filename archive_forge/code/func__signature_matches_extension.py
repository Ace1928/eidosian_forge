from __future__ import annotations
import os
import typing as ty
import numpy as np
from .arrayproxy import is_proxy
from .deprecated import deprecate_with_version
from .filebasedimages import ImageFileError
from .filename_parser import _stringify_path, splitext_addext
from .imageclasses import all_image_classes
from .openers import ImageOpener
def _signature_matches_extension(filename: FileSpec) -> tuple[bool, str]:
    """Check if signature aka magic number matches filename extension.

    Parameters
    ----------
    filename : str or os.PathLike
        Path to the file to check

    Returns
    -------
    matches : bool
       - `True` if the filename extension is not recognized (not .gz nor .bz2)
       - `True` if the magic number was successfully read and corresponds to
         the format indicated by the extension.
       - `False` otherwise.
    error_message : str
       An error message if opening the file failed or a mismatch is detected;
       the empty string otherwise.

    """
    signatures: dict[str, Signature] = {'.gz': {'signature': b'\x1f\x8b', 'format_name': 'gzip'}, '.bz2': {'signature': b'BZh', 'format_name': 'bzip2'}, '.zst': {'signature': b'(\xb5/\xfd', 'format_name': 'ztsd'}}
    filename = _stringify_path(filename)
    *_, ext = splitext_addext(filename)
    ext = ext.lower()
    if ext not in signatures:
        return (True, '')
    expected_signature = signatures[ext]['signature']
    try:
        with open(filename, 'rb') as fh:
            sniff = fh.read(len(expected_signature))
    except OSError:
        return (False, f'Could not read file: {filename}')
    if sniff.startswith(expected_signature):
        return (True, '')
    format_name = signatures[ext]['format_name']
    return (False, f'File {filename} is not a {format_name} file')