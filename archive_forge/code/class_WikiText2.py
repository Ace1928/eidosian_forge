import os
import zipfile
import shutil
import numpy as np
from . import _constants as C
from ...data import dataset
from ...utils import download, check_sha1, _get_repo_file_url
from ....contrib import text
from .... import nd, base
class WikiText2(_WikiText):
    """WikiText-2 word-level dataset for language modeling, from Salesforce research.

    From
    https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/

    License: Creative Commons Attribution-ShareAlike

    Each sample is a vector of length equal to the specified sequence length.
    At the end of each sentence, an end-of-sentence token '<eos>' is added.

    Parameters
    ----------
    root : str, default $MXNET_HOME/datasets/wikitext-2
        Path to temp folder for storing data.
    segment : str, default 'train'
        Dataset segment. Options are 'train', 'validation', 'test'.
    vocab : :class:`~mxnet.contrib.text.vocab.Vocabulary`, default None
        The vocabulary to use for indexing the text dataset.
        If None, a default vocabulary is created.
    seq_len : int, default 35
        The sequence length of each sample, regardless of the sentence boundary.

    """

    def __init__(self, root=os.path.join(base.data_dir(), 'datasets', 'wikitext-2'), segment='train', vocab=None, seq_len=35):
        self._archive_file = ('wikitext-2-v1.zip', '3c914d17d80b1459be871a5039ac23e752a53cbe')
        self._data_file = {'train': ('wiki.train.tokens', '863f29c46ef9d167fff4940ec821195882fe29d1'), 'validation': ('wiki.valid.tokens', '0418625c8b4da6e4b5c7a0b9e78d4ae8f7ee5422'), 'test': ('wiki.test.tokens', 'c7b8ce0aa086fb34dab808c5c49224211eb2b172')}
        self._segment = segment
        self._seq_len = seq_len
        super(WikiText2, self).__init__(root, 'wikitext-2', vocab)