from __future__ import absolute_import
import types
from . import Errors
class RE(object):
    """RE is the base class for regular expression constructors.
    The following operators are defined on REs:

         re1 + re2         is an RE which matches |re1| followed by |re2|
         re1 | re2         is an RE which matches either |re1| or |re2|
    """
    nullable = 1
    match_nl = 1
    str = None

    def build_machine(self, machine, initial_state, final_state, match_bol, nocase):
        """
        This method should add states to |machine| to implement this
        RE, starting at |initial_state| and ending at |final_state|.
        If |match_bol| is true, the RE must be able to match at the
        beginning of a line. If nocase is true, upper and lower case
        letters should be treated as equivalent.
        """
        raise NotImplementedError('%s.build_machine not implemented' % self.__class__.__name__)

    def build_opt(self, m, initial_state, c):
        """
        Given a state |s| of machine |m|, return a new state
        reachable from |s| on character |c| or epsilon.
        """
        s = m.new_state()
        initial_state.link_to(s)
        initial_state.add_transition(c, s)
        return s

    def __add__(self, other):
        return Seq(self, other)

    def __or__(self, other):
        return Alt(self, other)

    def __str__(self):
        if self.str:
            return self.str
        else:
            return self.calc_str()

    def check_re(self, num, value):
        if not isinstance(value, RE):
            self.wrong_type(num, value, 'Plex.RE instance')

    def check_string(self, num, value):
        if type(value) != type(''):
            self.wrong_type(num, value, 'string')

    def check_char(self, num, value):
        self.check_string(num, value)
        if len(value) != 1:
            raise Errors.PlexValueError('Invalid value for argument %d of Plex.%s.Expected a string of length 1, got: %s' % (num, self.__class__.__name__, repr(value)))

    def wrong_type(self, num, value, expected):
        if type(value) == types.InstanceType:
            got = '%s.%s instance' % (value.__class__.__module__, value.__class__.__name__)
        else:
            got = type(value).__name__
        raise Errors.PlexTypeError('Invalid type for argument %d of Plex.%s (expected %s, got %s' % (num, self.__class__.__name__, expected, got))