from ..base import traits, TraitedSpec, DynamicTraitedSpec, File, BaseInterface
from ..io import add_traits
def _append_entry(self, outputs, entry):
    for key, value in zip(self._outfields, entry):
        outputs[key].append(value)
    return outputs