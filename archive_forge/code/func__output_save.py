import collections
import gzip
import io
import logging
import struct
import numpy as np
def _output_save(fout, model):
    """
    Saves output layer of `model` to the binary stream `fout` containing a model in
    the Facebook's native fastText `.bin` format.

    Corresponding C++ fastText code:
    [DenseMatrix::save](https://github.com/facebookresearch/fastText/blob/master/src/densematrix.cc)

    Parameters
    ----------
    fout: writeable binary stream
        the model that contains the output layer to save
    model: gensim.models.fasttext.FastText
        saved model
    """
    if model.hs:
        hidden_output = model.syn1
    if model.negative:
        hidden_output = model.syn1neg
    hidden_n, hidden_dim = hidden_output.shape
    fout.write(struct.pack('@2q', hidden_n, hidden_dim))
    fout.write(hidden_output.tobytes())