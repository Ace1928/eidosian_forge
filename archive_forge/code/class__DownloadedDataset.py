import os
from ... import recordio, ndarray
class _DownloadedDataset(Dataset):
    """Base class for MNIST, cifar10, etc."""

    def __init__(self, root, transform):
        super(_DownloadedDataset, self).__init__()
        self._transform = transform
        self._data = None
        self._label = None
        root = os.path.expanduser(root)
        self._root = root
        if not os.path.isdir(root):
            os.makedirs(root)
        self._get_data()

    def __getitem__(self, idx):
        if self._transform is not None:
            return self._transform(self._data[idx], self._label[idx])
        return (self._data[idx], self._label[idx])

    def __len__(self):
        return len(self._label)

    def _get_data(self):
        raise NotImplementedError