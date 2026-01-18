import os
import gzip
import tarfile
import struct
import warnings
import numpy as np
from .. import dataset
from ...utils import download, check_sha1, _get_repo_file_url
from .... import nd, image, recordio, base
from .... import numpy as _mx_np  # pylint: disable=reimported
from ....util import is_np_array
def _read_batch(self, filename):
    with open(filename, 'rb') as fin:
        data = np.frombuffer(fin.read(), dtype=np.uint8).reshape(-1, 3072 + 2)
    return (data[:, 2:].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1), data[:, 0 + self._fine_label].astype(np.int32))