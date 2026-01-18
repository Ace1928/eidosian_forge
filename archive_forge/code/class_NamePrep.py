import stringprep
from encodings import idna
from itertools import chain
from unicodedata import ucd_3_2_0 as unicodedata
from zope.interface import Interface, implementer
from incremental import Version
from twisted.python.deprecate import deprecatedModuleAttribute
class NamePrep:
    """Implements preparation of internationalized domain names.

    This class implements preparing internationalized domain names using the
    rules defined in RFC 3491, section 4 (Conversion operations).

    We do not perform step 4 since we deal with unicode representations of
    domain names and do not convert from or to ASCII representations using
    punycode encoding. When such a conversion is needed, the C{idna} standard
    library provides the C{ToUnicode()} and C{ToASCII()} functions. Note that
    C{idna} itself assumes UseSTD3ASCIIRules to be false.

    The following steps are performed by C{prepare()}:

      - Split the domain name in labels at the dots (RFC 3490, 3.1)
      - Apply nameprep proper on each label (RFC 3491)
      - Enforce the restrictions on ASCII characters in host names by
        assuming STD3ASCIIRules to be true. (STD 3)
      - Rejoin the labels using the label separator U+002E (full stop).

    """
    prohibiteds = [chr(n) for n in chain(range(0, 44 + 1), range(46, 47 + 1), range(58, 64 + 1), range(91, 96 + 1), range(123, 127 + 1))]

    def prepare(self, string):
        result = []
        labels = idna.dots.split(string)
        if labels and len(labels[-1]) == 0:
            trailing_dot = '.'
            del labels[-1]
        else:
            trailing_dot = ''
        for label in labels:
            result.append(self.nameprep(label))
        return '.'.join(result) + trailing_dot

    def check_prohibiteds(self, string):
        for c in string:
            if c in self.prohibiteds:
                raise UnicodeError('Invalid character %s' % repr(c))

    def nameprep(self, label):
        label = idna.nameprep(label)
        self.check_prohibiteds(label)
        if label[0] == '-':
            raise UnicodeError('Invalid leading hyphen-minus')
        if label[-1] == '-':
            raise UnicodeError('Invalid trailing hyphen-minus')
        return label