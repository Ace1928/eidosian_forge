from typing import Callable, Optional, Tuple, cast
from ..config import registry
from ..model import Model
from ..types import Floats1d, Floats2d
from ..util import get_width
Compare input vectors according to the Cauchy similarity function proposed by
    Chen (2013). Primarily used within Siamese neural networks.
    