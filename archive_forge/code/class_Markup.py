from functools import reduce
import sys
from itertools import chain
import operator
import six
from genshi.compat import stringrepr
from genshi.util import stripentities, striptags
class Markup(six.text_type):
    """Marks a string as being safe for inclusion in HTML/XML output without
    needing to be escaped.
    """
    __slots__ = []

    def __add__(self, other):
        return Markup(six.text_type.__add__(self, escape(other)))

    def __radd__(self, other):
        return Markup(six.text_type.__add__(escape(other), self))

    def __mod__(self, args):
        if isinstance(args, dict):
            args = dict(zip(args.keys(), map(escape, args.values())))
        elif isinstance(args, (list, tuple)):
            args = tuple(map(escape, args))
        else:
            args = escape(args)
        return Markup(six.text_type.__mod__(self, args))

    def __mul__(self, num):
        return Markup(six.text_type.__mul__(self, num))
    __rmul__ = __mul__

    def __repr__(self):
        return '<%s %s>' % (type(self).__name__, six.text_type.__repr__(self))

    def join(self, seq, escape_quotes=True):
        """Return a `Markup` object which is the concatenation of the strings
        in the given sequence, where this `Markup` object is the separator
        between the joined elements.
        
        Any element in the sequence that is not a `Markup` instance is
        automatically escaped.
        
        :param seq: the sequence of strings to join
        :param escape_quotes: whether double quote characters in the elements
                              should be escaped
        :return: the joined `Markup` object
        :rtype: `Markup`
        :see: `escape`
        """
        escaped_items = [escape(item, quotes=escape_quotes) for item in seq]
        return Markup(six.text_type.join(self, escaped_items))

    @classmethod
    def escape(cls, text, quotes=True):
        """Create a Markup instance from a string and escape special characters
        it may contain (<, >, & and ").
        
        >>> escape('"1 < 2"')
        <Markup '&#34;1 &lt; 2&#34;'>
        
        If the `quotes` parameter is set to `False`, the " character is left
        as is. Escaping quotes is generally only required for strings that are
        to be used in attribute values.
        
        >>> escape('"1 < 2"', quotes=False)
        <Markup '"1 &lt; 2"'>
        
        :param text: the text to escape
        :param quotes: if ``True``, double quote characters are escaped in
                       addition to the other special characters
        :return: the escaped `Markup` string
        :rtype: `Markup`
        """
        if not text:
            return cls()
        if type(text) is cls:
            return text
        if hasattr(text, '__html__'):
            return cls(text.__html__())
        text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        if quotes:
            text = text.replace('"', '&#34;')
        return cls(text)

    def unescape(self):
        """Reverse-escapes &, <, >, and " and returns a `unicode` object.
        
        >>> Markup('1 &lt; 2').unescape()
        '1 < 2'
        
        :return: the unescaped string
        :rtype: `unicode`
        :see: `genshi.core.unescape`
        """
        if not self:
            return ''
        return six.text_type(self).replace('&#34;', '"').replace('&gt;', '>').replace('&lt;', '<').replace('&amp;', '&')

    def stripentities(self, keepxmlentities=False):
        """Return a copy of the text with any character or numeric entities
        replaced by the equivalent UTF-8 characters.
        
        If the `keepxmlentities` parameter is provided and evaluates to `True`,
        the core XML entities (``&amp;``, ``&apos;``, ``&gt;``, ``&lt;`` and
        ``&quot;``) are not stripped.
        
        :return: a `Markup` instance with entities removed
        :rtype: `Markup`
        :see: `genshi.util.stripentities`
        """
        return Markup(stripentities(self, keepxmlentities=keepxmlentities))

    def striptags(self):
        """Return a copy of the text with all XML/HTML tags removed.
        
        :return: a `Markup` instance with all tags removed
        :rtype: `Markup`
        :see: `genshi.util.striptags`
        """
        return Markup(striptags(self))