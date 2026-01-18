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
@register
class GloVe(_TokenEmbedding):
    """The GloVe word embedding.


    GloVe is an unsupervised learning algorithm for obtaining vector representations for words.
    Training is performed on aggregated global word-word co-occurrence statistics from a corpus, and
    the resulting representations showcase interesting linear substructures of the word vector
    space. (Source from https://nlp.stanford.edu/projects/glove/)

    References
    ----------

    GloVe: Global Vectors for Word Representation.
    Jeffrey Pennington, Richard Socher, and Christopher D. Manning.
    https://nlp.stanford.edu/pubs/glove.pdf

    Website:

    https://nlp.stanford.edu/projects/glove/

    To get the updated URLs to the externally hosted pre-trained token embedding
    files, visit https://nlp.stanford.edu/projects/glove/

    License for pre-trained embeddings:

        https://fedoraproject.org/wiki/Licensing/PDDL


    Parameters
    ----------
    pretrained_file_name : str, default 'glove.840B.300d.txt'
        The name of the pre-trained token embedding file.
    embedding_root : str, default $MXNET_HOME/embeddings
        The root directory for storing embedding-related files.
    init_unknown_vec : callback
        The callback used to initialize the embedding vector for the unknown token.
    vocabulary : :class:`~mxnet.contrib.text.vocab.Vocabulary`, default None
        It contains the tokens to index. Each indexed token will be associated with the loaded
        embedding vectors, such as loaded from a pre-trained token embedding file. If None, all the
        tokens from the loaded embedding vectors, such as loaded from a pre-trained token embedding
        file, will be indexed.
    """
    pretrained_archive_name_sha1 = C.GLOVE_PRETRAINED_FILE_SHA1
    pretrained_file_name_sha1 = C.GLOVE_PRETRAINED_ARCHIVE_SHA1

    @classmethod
    def _get_download_file_name(cls, pretrained_file_name):
        src_archive = {archive.split('.')[1]: archive for archive in GloVe.pretrained_archive_name_sha1.keys()}
        archive = src_archive[pretrained_file_name.split('.')[1]]
        return archive

    def __init__(self, pretrained_file_name='glove.840B.300d.txt', embedding_root=os.path.join(base.data_dir(), 'embeddings'), init_unknown_vec=nd.zeros, vocabulary=None, **kwargs):
        GloVe._check_pretrained_file_names(pretrained_file_name)
        super(GloVe, self).__init__(**kwargs)
        pretrained_file_path = GloVe._get_pretrained_file(embedding_root, pretrained_file_name)
        self._load_embedding(pretrained_file_path, ' ', init_unknown_vec)
        if vocabulary is not None:
            self._build_embedding_for_vocabulary(vocabulary)