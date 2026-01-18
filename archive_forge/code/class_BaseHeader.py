from types import MappingProxyType
from email import utils
from email import errors
from email import _header_value_parser as parser
class BaseHeader(str):
    """Base class for message headers.

    Implements generic behavior and provides tools for subclasses.

    A subclass must define a classmethod named 'parse' that takes an unfolded
    value string and a dictionary as its arguments.  The dictionary will
    contain one key, 'defects', initialized to an empty list.  After the call
    the dictionary must contain two additional keys: parse_tree, set to the
    parse tree obtained from parsing the header, and 'decoded', set to the
    string value of the idealized representation of the data from the value.
    (That is, encoded words are decoded, and values that have canonical
    representations are so represented.)

    The defects key is intended to collect parsing defects, which the message
    parser will subsequently dispose of as appropriate.  The parser should not,
    insofar as practical, raise any errors.  Defects should be added to the
    list instead.  The standard header parsers register defects for RFC
    compliance issues, for obsolete RFC syntax, and for unrecoverable parsing
    errors.

    The parse method may add additional keys to the dictionary.  In this case
    the subclass must define an 'init' method, which will be passed the
    dictionary as its keyword arguments.  The method should use (usually by
    setting them as the value of similarly named attributes) and remove all the
    extra keys added by its parse method, and then use super to call its parent
    class with the remaining arguments and keywords.

    The subclass should also make sure that a 'max_count' attribute is defined
    that is either None or 1. XXX: need to better define this API.

    """

    def __new__(cls, name, value):
        kwds = {'defects': []}
        cls.parse(value, kwds)
        if utils._has_surrogates(kwds['decoded']):
            kwds['decoded'] = utils._sanitize(kwds['decoded'])
        self = str.__new__(cls, kwds['decoded'])
        del kwds['decoded']
        self.init(name, **kwds)
        return self

    def init(self, name, *, parse_tree, defects):
        self._name = name
        self._parse_tree = parse_tree
        self._defects = defects

    @property
    def name(self):
        return self._name

    @property
    def defects(self):
        return tuple(self._defects)

    def __reduce__(self):
        return (_reconstruct_header, (self.__class__.__name__, self.__class__.__bases__, str(self)), self.__getstate__())

    @classmethod
    def _reconstruct(cls, value):
        return str.__new__(cls, value)

    def fold(self, *, policy):
        """Fold header according to policy.

        The parsed representation of the header is folded according to
        RFC5322 rules, as modified by the policy.  If the parse tree
        contains surrogateescaped bytes, the bytes are CTE encoded using
        the charset 'unknown-8bit".

        Any non-ASCII characters in the parse tree are CTE encoded using
        charset utf-8. XXX: make this a policy setting.

        The returned value is an ASCII-only string possibly containing linesep
        characters, and ending with a linesep character.  The string includes
        the header name and the ': ' separator.

        """
        header = parser.Header([parser.HeaderLabel([parser.ValueTerminal(self.name, 'header-name'), parser.ValueTerminal(':', 'header-sep')])])
        if self._parse_tree:
            header.append(parser.CFWSList([parser.WhiteSpaceTerminal(' ', 'fws')]))
        header.append(self._parse_tree)
        return header.fold(policy=policy)