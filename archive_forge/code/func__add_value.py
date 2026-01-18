from contextlib import suppress
from itemadapter import ItemAdapter
from parsel.utils import extract_regex, flatten
from itemloaders.common import wrap_loader_context
from itemloaders.processors import Identity
from itemloaders.utils import arg_to_iter
def _add_value(self, field_name, value):
    value = arg_to_iter(value)
    processed_value = self._process_input_value(field_name, value)
    if processed_value:
        self._values.setdefault(field_name, [])
        self._values[field_name] += arg_to_iter(processed_value)