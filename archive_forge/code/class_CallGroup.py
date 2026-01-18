import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
class CallGroup(RegexBase):

    def __init__(self, info, group, position):
        RegexBase.__init__(self)
        self.info = info
        self.group = group
        self.position = position
        self._key = (self.__class__, self.group)

    def fix_groups(self, pattern, reverse, fuzzy):
        try:
            self.group = int(self.group)
        except ValueError:
            try:
                self.group = self.info.group_index[self.group]
            except KeyError:
                raise error('invalid group reference', pattern, self.position)
        if not 0 <= self.group <= self.info.group_count:
            raise error('unknown group', pattern, self.position)
        if self.group > 0 and self.info.open_group_count[self.group] > 1:
            raise error('ambiguous group reference', pattern, self.position)
        self.info.group_calls.append((self, reverse, fuzzy))
        self._key = (self.__class__, self.group)

    def remove_captures(self):
        raise error('group reference not allowed', pattern, self.position)

    def _compile(self, reverse, fuzzy):
        return [(OP.GROUP_CALL, self.call_ref)]

    def dump(self, indent, reverse):
        print('{}GROUP_CALL {}'.format(INDENT * indent, self.group))

    def __eq__(self, other):
        return type(self) is type(other) and self.group == other.group

    def max_width(self):
        return UNLIMITED

    def __del__(self):
        self.info = None