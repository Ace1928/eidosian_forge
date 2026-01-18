from ..base import traits, TraitedSpec, DynamicTraitedSpec, File, BaseInterface
from ..io import add_traits
def _outputs(self):
    return self._add_output_traits(super(CSVReader, self)._outputs())