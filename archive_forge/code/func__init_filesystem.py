import os
import re
import urllib
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional
from ray._private.client_mode_hook import client_mode_hook
from ray._private.utils import _add_creatable_buckets_param_if_s3_uri, load_class
from ray._private.auto_init_hook import wrap_auto_init
def _init_filesystem(create_valid_file: bool=False, check_valid_file: bool=True):
    """Fully initialize the filesystem at the given storage URI."""
    global _filesystem, _storage_prefix, _storage_uri
    assert _filesystem is None, 'Init can only be called once.'
    if not _storage_uri:
        raise RuntimeError('No storage URI has been configured for the cluster. Specify a storage URI via `ray.init(storage=<uri>)` or `ray start --head --storage=<uri>`')
    import pyarrow.fs
    parsed_uri = urllib.parse.urlparse(_storage_uri.replace('\\', '/'))
    if parsed_uri.scheme == 'custom':
        fs_creator = _load_class(parsed_uri.netloc)
        _filesystem, _storage_prefix = fs_creator(parsed_uri.path)
    else:
        _storage_uri = _add_creatable_buckets_param_if_s3_uri(_storage_uri)
        _filesystem, _storage_prefix = pyarrow.fs.FileSystem.from_uri(_storage_uri)
    if os.name == 'nt':
        if re.match('^//[A-Za-z]/.*', _storage_prefix):
            _storage_prefix = _storage_prefix[2] + ':' + _storage_prefix[4:]
    valid_file = _storage_prefix + '/_valid'
    if create_valid_file:
        _filesystem.create_dir(_storage_prefix)
        with _filesystem.open_output_stream(valid_file):
            pass
    if check_valid_file:
        valid = _filesystem.get_file_info([valid_file])[0]
        if valid.type == pyarrow.fs.FileType.NotFound:
            raise RuntimeError('Unable to initialize storage: {} file created during init not found. Check that configured cluster storage path is readable from all worker nodes of the cluster.'.format(valid_file))
    return (_filesystem, _storage_prefix)