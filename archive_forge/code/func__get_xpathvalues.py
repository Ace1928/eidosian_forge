from contextlib import suppress
from itemadapter import ItemAdapter
from parsel.utils import extract_regex, flatten
from itemloaders.common import wrap_loader_context
from itemloaders.processors import Identity
from itemloaders.utils import arg_to_iter
def _get_xpathvalues(self, xpaths, **kw):
    self._check_selector_method()
    xpaths = arg_to_iter(xpaths)
    return flatten((self.selector.xpath(xpath, **kw).getall() for xpath in xpaths))