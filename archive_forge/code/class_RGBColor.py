import collections
class RGBColor(collections.namedtuple('RGBColor', ['red', 'green', 'blue'])):
    """Named tuple for an RGB color definition."""

    def __str__(self):
        return '#{0:02x}{1:02x}{2:02x}'.format(*self)