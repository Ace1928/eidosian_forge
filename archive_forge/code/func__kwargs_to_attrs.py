import six
from genshi.compat import numeric_types
from genshi.core import Attrs, Markup, Namespace, QName, Stream, \
def _kwargs_to_attrs(kwargs):
    attrs = []
    names = set()
    for name, value in kwargs.items():
        name = name.rstrip('_').replace('_', '-')
        if value is not None and name not in names:
            attrs.append((QName(name), six.text_type(value)))
            names.add(name)
    return Attrs(attrs)