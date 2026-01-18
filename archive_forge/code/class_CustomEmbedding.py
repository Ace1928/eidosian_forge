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
class CustomEmbedding(_TokenEmbedding):
    """User-defined token embedding.

    This is to load embedding vectors from a user-defined pre-trained text embedding file.

    Denote by '[ed]' the argument `elem_delim`. Denote by [v_ij] the j-th element of the token
    embedding vector for [token_i], the expected format of a custom pre-trained token embedding file
    is:

    '[token_1][ed][v_11][ed][v_12][ed]...[ed][v_1k]\\\\n[token_2][ed][v_21][ed][v_22][ed]...[ed]
    [v_2k]\\\\n...'

    where k is the length of the embedding vector `vec_len`.


    Parameters
    ----------
    pretrained_file_path : str
        The path to the custom pre-trained token embedding file.
    elem_delim : str, default ' '
        The delimiter for splitting a token and every embedding vector element value on the same
        line of the custom pre-trained token embedding file.
    encoding : str, default 'utf8'
        The encoding scheme for reading the custom pre-trained token embedding file.
    init_unknown_vec : callback
        The callback used to initialize the embedding vector for the unknown token.
    vocabulary : :class:`~mxnet.contrib.text.vocab.Vocabulary`, default None
        It contains the tokens to index. Each indexed token will be associated with the loaded
        embedding vectors, such as loaded from a pre-trained token embedding file. If None, all the
        tokens from the loaded embedding vectors, such as loaded from a pre-trained token embedding
        file, will be indexed.
    """

    def __init__(self, pretrained_file_path, elem_delim=' ', encoding='utf8', init_unknown_vec=nd.zeros, vocabulary=None, **kwargs):
        super(CustomEmbedding, self).__init__(**kwargs)
        self._load_embedding(pretrained_file_path, elem_delim, init_unknown_vec, encoding)
        if vocabulary is not None:
            self._build_embedding_for_vocabulary(vocabulary)