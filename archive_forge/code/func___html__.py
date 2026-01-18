import six
from genshi.compat import numeric_types
from genshi.core import Attrs, Markup, Namespace, QName, Stream, \
def __html__(self):
    return Markup(self.generate())