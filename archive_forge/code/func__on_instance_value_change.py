from functools import partial
import numpy as np
from traitlets import Union, Instance, Undefined, TraitError
from .serializers import data_union_serialization
from .traits import NDArray
from .widgets import NDArrayWidget, NDArrayBase, NDArraySource
def _on_instance_value_change(self, change):
    inst = change['owner']
    if isinstance(change['old'], NDArrayWidget):
        f = self._registered_validators.pop(inst, None)
        if f is not None:
            change['old']._instance_validators.remove(f)
        f = self._registered_observer.pop(inst, None)
        if f is not None:
            change['old'].unobserve(f)
    if isinstance(change['new'], NDArrayWidget):
        f = partial(self._validate_child, inst)
        self._registered_validators[inst] = f
        change['new']._instance_validators.add(f)
        f = partial(self._on_widget_array_change, inst)
        self._registered_observer[inst] = f
        change['new'].observe(f, 'array')