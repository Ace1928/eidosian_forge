from typing import List, Optional, Union
from thinc.api import Model
from thinc.types import Floats2d
from spacy.attrs import intify_attr
from spacy.errors import Errors
from spacy.tokens import Doc
from spacy.util import registry
def MaxoutWindowEncoder_v1(width: int, window_size: int, maxout_pieces: int, depth: int) -> Model[Floats2d, Floats2d]:
    """Encode context using convolutions with maxout activation, layer
    normalization and residual connections.

    width (int): The input and output width. These are required to be the same,
        to allow residual connections. This value will be determined by the
        width of the inputs. Recommended values are between 64 and 300.
    window_size (int): The number of words to concatenate around each token
        to construct the convolution. Recommended value is 1.
    maxout_pieces (int): The number of maxout pieces to use. Recommended
        values are 2 or 3.
    depth (int): The number of convolutional layers. Recommended value is 4.
    """
    Maxout = registry.get('layers', 'Maxout.v1')
    chain = registry.get('layers', 'chain.v1')
    expand_window = registry.get('layers', 'expand_window.v1')
    clone = registry.get('layers', 'clone.v1')
    residual = registry.get('layers', 'residual.v1')
    cnn = chain(expand_window(window_size=window_size), Maxout(nO=width, nI=width * (window_size * 2 + 1), nP=maxout_pieces, dropout=0.0, normalize=True))
    model = clone(residual(cnn), depth)
    model.set_dim('nO', width)
    model.attrs['receptive_field'] = window_size * depth
    return model