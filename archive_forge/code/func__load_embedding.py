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
def _load_embedding(self, pretrained_file_path, elem_delim, init_unknown_vec, encoding='utf8'):
    """Load embedding vectors from the pre-trained token embedding file.


        For every unknown token, if its representation `self.unknown_token` is encountered in the
        pre-trained token embedding file, index 0 of `self.idx_to_vec` maps to the pre-trained token
        embedding vector loaded from the file; otherwise, index 0 of `self.idx_to_vec` maps to the
        text embedding vector initialized by `init_unknown_vec`.

        If a token is encountered multiple times in the pre-trained text embedding file, only the
        first-encountered token embedding vector will be loaded and the rest will be skipped.
        """
    pretrained_file_path = os.path.expanduser(pretrained_file_path)
    if not os.path.isfile(pretrained_file_path):
        raise ValueError('`pretrained_file_path` must be a valid path to the pre-trained token embedding file.')
    logging.info('Loading pre-trained token embedding vectors from %s', pretrained_file_path)
    vec_len = None
    all_elems = []
    tokens = set()
    loaded_unknown_vec = None
    line_num = 0
    with open(pretrained_file_path, 'r', encoding=encoding) as f:
        for line in f:
            line_num += 1
            elems = line.rstrip().split(elem_delim)
            assert len(elems) > 1, 'At line %d of the pre-trained text embedding file: the data format of the pre-trained token embedding file %s is unexpected.' % (line_num, pretrained_file_path)
            token, elems = (elems[0], [float(i) for i in elems[1:]])
            if token == self.unknown_token and loaded_unknown_vec is None:
                loaded_unknown_vec = elems
                tokens.add(self.unknown_token)
            elif token in tokens:
                warnings.warn('At line %d of the pre-trained token embedding file: the embedding vector for token %s has been loaded and a duplicate embedding for the  same token is seen and skipped.' % (line_num, token))
            elif len(elems) == 1:
                warnings.warn('At line %d of the pre-trained text embedding file: token %s with 1-dimensional vector %s is likely a header and is skipped.' % (line_num, token, elems))
            else:
                if vec_len is None:
                    vec_len = len(elems)
                    all_elems.extend([0] * vec_len)
                else:
                    assert len(elems) == vec_len, 'At line %d of the pre-trained token embedding file: the dimension of token %s is %d but the dimension of previous tokens is %d. Dimensions of all the tokens must be the same.' % (line_num, token, len(elems), vec_len)
                all_elems.extend(elems)
                self._idx_to_token.append(token)
                self._token_to_idx[token] = len(self._idx_to_token) - 1
                tokens.add(token)
    self._vec_len = vec_len
    array_fn = _mx_np.array if is_np_array() else nd.array
    self._idx_to_vec = array_fn(all_elems).reshape((-1, self.vec_len))
    if loaded_unknown_vec is None:
        init_val = init_unknown_vec(shape=self.vec_len)
        self._idx_to_vec[C.UNKNOWN_IDX] = init_val.as_np_ndarray() if is_np_array() else init_val
    else:
        self._idx_to_vec[C.UNKNOWN_IDX] = array_fn(loaded_unknown_vec)