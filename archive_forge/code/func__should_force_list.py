from xml.sax.saxutils import XMLGenerator
from xml.sax.xmlreader import AttributesImpl
import platform
from inspect import isgenerator
def _should_force_list(self, key, value):
    if not self.force_list:
        return False
    if isinstance(self.force_list, bool):
        return self.force_list
    try:
        return key in self.force_list
    except TypeError:
        return self.force_list(self.path[:-1], key, value)