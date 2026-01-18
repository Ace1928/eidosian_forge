import os
import zipfile
import shutil
import numpy as np
from . import _constants as C
from ...data import dataset
from ...utils import download, check_sha1, _get_repo_file_url
from ....contrib import text
from .... import nd, base
class WikiText103(_WikiText):
    """WikiText-103 word-level dataset for language modeling, from Salesforce research.

    From
    https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/

    License: Creative Commons Attribution-ShareAlike

    Each sample is a vector of length equal to the specified sequence length.
    At the end of each sentence, an end-of-sentence token '<eos>' is added.

    Parameters
    ----------
    root : str, default $MXNET_HOME/datasets/wikitext-103
        Path to temp folder for storing data.
    segment : str, default 'train'
        Dataset segment. Options are 'train', 'validation', 'test'.
    vocab : :class:`~mxnet.contrib.text.vocab.Vocabulary`, default None
        The vocabulary to use for indexing the text dataset.
        If None, a default vocabulary is created.
    seq_len : int, default 35
        The sequence length of each sample, regardless of the sentence boundary.
    """

    def __init__(self, root=os.path.join(base.data_dir(), 'datasets', 'wikitext-103'), segment='train', vocab=None, seq_len=35):
        self._archive_file = ('wikitext-103-v1.zip', '0aec09a7537b58d4bb65362fee27650eeaba625a')
        self._data_file = {'train': ('wiki.train.tokens', 'b7497e2dfe77e72cfef5e3dbc61b7b53712ac211'), 'validation': ('wiki.valid.tokens', 'c326ac59dc587676d58c422eb8a03e119582f92b'), 'test': ('wiki.test.tokens', '8a5befc548865cec54ed4273cf87dbbad60d1e47')}
        self._segment = segment
        self._seq_len = seq_len
        super(WikiText103, self).__init__(root, 'wikitext-103', vocab)