from collections import namedtuple
from distutils.version import LooseVersion
import io
import operator
import tempfile
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torchtext
from torchtext.data.utils import get_tokenizer
from torchtext.utils import download_from_url, extract_archive
def _batchify(data, batch_size):
    data = torch.tensor(data)
    nbatch = data.size(0) // batch_size
    data = data.narrow(0, 0, nbatch * batch_size)
    data = data.view(batch_size, -1).t().contiguous()
    return data