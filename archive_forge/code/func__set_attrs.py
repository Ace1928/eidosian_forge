import sys, os
import textwrap
def _set_attrs(self, attrs):
    for attr in self.ATTRS:
        if attr in attrs:
            setattr(self, attr, attrs[attr])
            del attrs[attr]
        elif attr == 'default':
            setattr(self, attr, NO_DEFAULT)
        else:
            setattr(self, attr, None)
    if attrs:
        attrs = sorted(attrs.keys())
        raise OptionError('invalid keyword arguments: %s' % ', '.join(attrs), self)