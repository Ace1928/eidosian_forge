from contextlib import suppress
from itemadapter import ItemAdapter
from parsel.utils import extract_regex, flatten
from itemloaders.common import wrap_loader_context
from itemloaders.processors import Identity
from itemloaders.utils import arg_to_iter
def _check_selector_method(self):
    if self.selector is None:
        raise RuntimeError('To use XPath or CSS selectors, %s must be instantiated with a selector' % self.__class__.__name__)