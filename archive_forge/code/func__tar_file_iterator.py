import fnmatch
import io
import re
import tarfile
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union
from ray.data.block import BlockAccessor
from ray.data.datasource.file_based_datasource import FileBasedDatasource
from ray.util.annotations import PublicAPI
def _tar_file_iterator(fileobj: Any, fileselect: Optional[Union[bool, callable, list]]=None, filerename: Optional[Union[bool, callable, list]]=None, verbose_open: bool=False, meta: dict=None):
    """Iterate over tar file, yielding filename, content pairs for the given tar stream.

    Args:
        fileobj: file object
        fileselect: patterns or function selecting
            files to be selected
        meta: metadata to be added to each sample
    """
    meta = meta or {}
    stream = tarfile.open(fileobj=fileobj, mode='r|*')
    if verbose_open:
        print(f'start {meta}')
    for tarinfo in stream:
        fname = tarinfo.name
        if not tarinfo.isreg() or fname is None:
            continue
        data = stream.extractfile(tarinfo).read()
        fname = _apply_list(filerename, fname)
        assert isinstance(fname, str)
        if not _check_suffix(fname, fileselect):
            continue
        result = dict(fname=fname, data=data)
        yield result
    if verbose_open:
        print(f'done {meta}')