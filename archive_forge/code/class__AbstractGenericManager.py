from jedi import debug
from jedi.cache import memoize_method
from jedi.inference.utils import to_tuple
from jedi.inference.base_value import ValueSet, NO_VALUES
from jedi.inference.value.iterable import SequenceLiteralValue
from jedi.inference.helpers import is_string
class _AbstractGenericManager:

    def get_index_and_execute(self, index):
        try:
            return self[index].execute_annotation()
        except IndexError:
            debug.warning('No param #%s found for annotation %s', index, self)
            return NO_VALUES

    def get_type_hint(self):
        return '[%s]' % ', '.join((t.get_type_hint(add_class_info=False) for t in self.to_tuple()))