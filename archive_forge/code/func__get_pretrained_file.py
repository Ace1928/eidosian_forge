import logging
import os
import tarfile
import warnings
import zipfile
from . import _constants as C
from . import vocab
from ... import ndarray as nd
from ... import registry
from ... import base
from ...util import is_np_array
from ... import numpy as _mx_np
from ... import numpy_extension as _mx_npx
@classmethod
def _get_pretrained_file(cls, embedding_root, pretrained_file_name):
    from ...gluon.utils import check_sha1, download
    embedding_cls = cls.__name__.lower()
    embedding_root = os.path.expanduser(embedding_root)
    url = cls._get_pretrained_file_url(pretrained_file_name)
    embedding_dir = os.path.join(embedding_root, embedding_cls)
    pretrained_file_path = os.path.join(embedding_dir, pretrained_file_name)
    downloaded_file = os.path.basename(url)
    downloaded_file_path = os.path.join(embedding_dir, downloaded_file)
    expected_file_hash = cls.pretrained_file_name_sha1[pretrained_file_name]
    if hasattr(cls, 'pretrained_archive_name_sha1'):
        expected_downloaded_hash = cls.pretrained_archive_name_sha1[downloaded_file]
    else:
        expected_downloaded_hash = expected_file_hash
    if not os.path.exists(pretrained_file_path) or not check_sha1(pretrained_file_path, expected_file_hash):
        download(url, downloaded_file_path, sha1_hash=expected_downloaded_hash)
        ext = os.path.splitext(downloaded_file)[1]
        if ext == '.zip':
            with zipfile.ZipFile(downloaded_file_path, 'r') as zf:
                zf.extractall(embedding_dir)
        elif ext == '.gz':
            with tarfile.open(downloaded_file_path, 'r:gz') as tar:
                tar.extractall(path=embedding_dir)
    return pretrained_file_path