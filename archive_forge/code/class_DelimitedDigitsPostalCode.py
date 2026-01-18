import re
from .api import FancyValidator
from .compound import Any
from .validators import Regex, Invalid, _
class DelimitedDigitsPostalCode(Regex):
    """
    Abstraction of common postal code formats, such as 55555, 55-555 etc.
    With constant amount of digits. By providing a single digit as partition
    you can obtain a trivial 'x digits' postal code validator.

    For flexibility, input may use additional delimiters or delimters in a
    bad position. Only the minimum (or if strict, exact) number of digits has
    to be provided.

    ::

        >>> german = DelimitedDigitsPostalCode(5)
        >>> german.to_python('55555')
        '55555'
        >>> german.to_python('55 55-5')
        '55555'
        >>> german.to_python('5555')
        Traceback (most recent call last):
            ...
        Invalid: Please enter a zip code (5 digits)
        >>> polish = DelimitedDigitsPostalCode([2, 3], '-')
        >>> polish.to_python('55555')
        '55-555'
        >>> polish.to_python('55-555')
        '55-555'
        >>> polish.to_python('555-55')
        '55-555'
        >>> polish.to_python('5555')
        Traceback (most recent call last):
            ...
        Invalid: Please enter a zip code (nn-nnn)
        >>> nicaragua = DelimitedDigitsPostalCode([3, 3, 1], '-')
        >>> nicaragua.to_python('5554443')
        '555-444-3'
        >>> nicaragua.to_python('555-4443')
        '555-444-3'
        >>> nicaragua.to_python('5555')
        Traceback (most recent call last):
            ...
        Invalid: Please enter a zip code (nnn-nnn-n)
    """
    strip = True

    def assembly_formatstring(self, partition_lengths, delimiter):
        if len(partition_lengths) == 1:
            return _('%d digits') % partition_lengths[0]
        return delimiter.join(('n' * n for n in partition_lengths))

    def assembly_grouping(self, partition_lengths, delimiter):
        digit_groups = ['%s' * length for length in partition_lengths]
        return delimiter.join(digit_groups)

    def assembly_regex(self, partition_lengths, delimiter, strict):
        regex = '\\D*(\\d)\\D*' * sum(partition_lengths)
        if strict:
            regex = '^' + regex + '$'
        return regex

    def __init__(self, partition_lengths, delimiter=None, strict=False, *args, **kw):
        if isinstance(partition_lengths, int):
            partition_lengths = [partition_lengths]
        if not delimiter:
            delimiter = ''
        self.format = self.assembly_formatstring(partition_lengths, delimiter)
        self.grouping = self.assembly_grouping(partition_lengths, delimiter)
        self.regex = self.assembly_regex(partition_lengths, delimiter, strict)
        self.partition_lengths, self.delimiter = (partition_lengths, delimiter)
        Regex.__init__(self, *args, **kw)
    messages = dict(invalid=_('Please enter a zip code (%(format)s)'))

    def _convert_to_python(self, value, state):
        self.assert_string(value, state)
        match = self.regex.search(value)
        if not match:
            raise Invalid(self.message('invalid', state, format=self.format), value, state)
        return self.grouping % match.groups()