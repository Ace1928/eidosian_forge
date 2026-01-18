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
class ImageFolderDataset(dataset.Dataset):
    """A dataset for loading image files stored in a folder structure.

    like::

        root/car/0001.jpg
        root/car/xxxa.jpg
        root/car/yyyb.jpg
        root/bus/123.jpg
        root/bus/023.jpg
        root/bus/wwww.jpg

    Parameters
    ----------
    root : str
        Path to root directory.
    flag : {0, 1}, default 1
        If 0, always convert loaded images to greyscale (1 channel).
        If 1, always convert loaded images to colored (3 channels).
    transform : callable, default None
        A function that takes data and label and transforms them::

            transform = lambda data, label: (data.astype(np.float32)/255, label)

    Attributes
    ----------
    synsets : list
        List of class names. `synsets[i]` is the name for the integer label `i`
    items : list of tuples
        List of all images in (filename, label) pairs.
    """

    def __init__(self, root, flag=1, transform=None):
        self._root = os.path.expanduser(root)
        self._flag = flag
        self._transform = transform
        self._exts = ['.jpg', '.jpeg', '.png']
        self._list_images(self._root)

    def _list_images(self, root):
        self.synsets = []
        self.items = []
        for folder in sorted(os.listdir(root)):
            path = os.path.join(root, folder)
            if not os.path.isdir(path):
                warnings.warn('Ignoring %s, which is not a directory.' % path, stacklevel=3)
                continue
            label = len(self.synsets)
            self.synsets.append(folder)
            for filename in sorted(os.listdir(path)):
                filename = os.path.join(path, filename)
                ext = os.path.splitext(filename)[1]
                if ext.lower() not in self._exts:
                    warnings.warn('Ignoring %s of type %s. Only support %s' % (filename, ext, ', '.join(self._exts)))
                    continue
                self.items.append((filename, label))

    def __getitem__(self, idx):
        img = image.imread(self.items[idx][0], self._flag)
        label = self.items[idx][1]
        if self._transform is not None:
            return self._transform(img, label)
        return (img, label)

    def __len__(self):
        return len(self.items)