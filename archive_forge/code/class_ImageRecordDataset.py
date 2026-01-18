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
class ImageRecordDataset(dataset.RecordFileDataset):
    """A dataset wrapping over a RecordIO file containing images.

    Each sample is an image and its corresponding label.

    Parameters
    ----------
    filename : str
        Path to rec file.
    flag : {0, 1}, default 1
        If 0, always convert images to greyscale.         If 1, always convert images to colored (RGB).
    transform : function, default None
        A user defined callback that transforms each sample. For example::

            transform=lambda data, label: (data.astype(np.float32)/255, label)

    """

    def __init__(self, filename, flag=1, transform=None):
        super(ImageRecordDataset, self).__init__(filename)
        self._flag = flag
        self._transform = transform

    def __getitem__(self, idx):
        record = super(ImageRecordDataset, self).__getitem__(idx)
        header, img = recordio.unpack(record)
        if self._transform is not None:
            return self._transform(image.imdecode(img, self._flag), header.label)
        return (image.imdecode(img, self._flag), header.label)