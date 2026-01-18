from tensorflow.python.framework import composite_tensor
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util import nest
Flattens `inputs` but don't expand `ResourceVariable`s.