from contextlib import suppress
from itemadapter import ItemAdapter
from parsel.utils import extract_regex, flatten
from itemloaders.common import wrap_loader_context
from itemloaders.processors import Identity
from itemloaders.utils import arg_to_iter
def _get_item_field_attr(self, field_name, key, default=None):
    field_meta = ItemAdapter(self.item).get_field_meta(field_name)
    return field_meta.get(key, default)