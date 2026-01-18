import fnmatch
import io
import re
import tarfile
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union
from ray.data.block import BlockAccessor
from ray.data.datasource.file_based_datasource import FileBasedDatasource
from ray.util.annotations import PublicAPI
def _group_by_keys(data: List[Dict[str, Any]], keys: callable=_base_plus_ext, suffixes: Optional[Union[list, callable]]=None, meta: dict=None):
    """Return function over iterator that groups key, value pairs into samples.

    Args:
        data: iterator over key, value pairs
        keys: function that returns key, suffix for a given key
        suffixes: list of suffixes to be included in the sample
        meta: metadata to be added to each sample
    """
    meta = meta or {}
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        fname, value = (filesample['fname'], filesample['data'])
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if current_sample is None or prefix != current_sample['__key__']:
            if _valid_sample(current_sample):
                current_sample.update(meta)
                yield current_sample
            current_sample = dict(__key__=prefix)
            if '__url__' in filesample:
                current_sample['__url__'] = filesample['__url__']
        if suffix in current_sample:
            raise ValueError(f'{fname}: duplicate file name in tar file ' + f'{suffix} {current_sample.keys()}')
        if suffixes is None or _check_suffix(suffix, suffixes):
            current_sample[suffix] = value
    if _valid_sample(current_sample):
        current_sample.update(meta)
        yield current_sample