from typing import List, Optional, Union, cast
from thinc.api import (
from thinc.types import Floats2d, Ints1d, Ints2d, Ragged
from ...attrs import intify_attr
from ...errors import Errors
from ...ml import _character_embed
from ...pipeline.tok2vec import Tok2VecListener
from ...tokens import Doc
from ...util import registry
from ..featureextractor import FeatureExtractor
from ..staticvectors import StaticVectors
@registry.architectures('spacy.MishWindowEncoder.v2')
def MishWindowEncoder(width: int, window_size: int, depth: int) -> Model[List[Floats2d], List[Floats2d]]:
    """Encode context using convolutions with mish activation, layer
    normalization and residual connections.

    width (int): The input and output width. These are required to be the same,
        to allow residual connections. This value will be determined by the
        width of the inputs. Recommended values are between 64 and 300.
    window_size (int): The number of words to concatenate around each token
        to construct the convolution. Recommended value is 1.
    depth (int): The number of convolutional layers. Recommended value is 4.
    """
    cnn = chain(expand_window(window_size=window_size), Mish(nO=width, nI=width * (window_size * 2 + 1), dropout=0.0, normalize=True))
    model = clone(residual(cnn), depth)
    model.set_dim('nO', width)
    return with_array(model)