import numpy as np
from . import check_consistent_length, check_matplotlib_support
from ._response import _get_response_values_binary
from .multiclass import type_of_target
from .validation import _check_pos_label_consistency
def _validate_plot_params(self, *, ax=None, name=None):
    check_matplotlib_support(f'{self.__class__.__name__}.plot')
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots()
    name = self.estimator_name if name is None else name
    return (ax, ax.figure, name)