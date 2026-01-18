from collections import OrderedDict, defaultdict
import warnings
import numpy as np
from ..base import mx_real_t, MXNetError
from .. import symbol, ndarray, initializer, context
from ..context import Context, cpu
from .. import autograd
from .utils import _indent, _brief_print_list, shape_is_known
from ..util import is_np_shape, is_np_array
from .. import numpy as _mx_np  # pylint: disable=reimported
def _set_trainer(self, trainer):
    """ Set the trainer this parameter is associated with. """
    if self._stype != 'default' and self._trainer and trainer and (self._trainer is not trainer):
        raise RuntimeError("Failed to set the trainer for Parameter '%s' because it was already set. More than one trainers for a %s Parameter is not supported." % (self.name, self._stype))
    self._trainer = trainer