import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
class RefGroup(RegexBase):
    _opcode = {(NOCASE, False): OP.REF_GROUP, (IGNORECASE, False): OP.REF_GROUP_IGN, (FULLCASE, False): OP.REF_GROUP, (FULLIGNORECASE, False): OP.REF_GROUP_FLD, (NOCASE, True): OP.REF_GROUP_REV, (IGNORECASE, True): OP.REF_GROUP_IGN_REV, (FULLCASE, True): OP.REF_GROUP_REV, (FULLIGNORECASE, True): OP.REF_GROUP_FLD_REV}

    def __init__(self, info, group, position, case_flags=NOCASE):
        RegexBase.__init__(self)
        self.info = info
        self.group = group
        self.position = position
        self.case_flags = CASE_FLAGS_COMBINATIONS[case_flags]
        self._key = (self.__class__, self.group, self.case_flags)

    def fix_groups(self, pattern, reverse, fuzzy):
        try:
            self.group = int(self.group)
        except ValueError:
            try:
                self.group = self.info.group_index[self.group]
            except KeyError:
                raise error('unknown group', pattern, self.position)
        if not 1 <= self.group <= self.info.group_count:
            raise error('invalid group reference', pattern, self.position)
        self._key = (self.__class__, self.group, self.case_flags)

    def remove_captures(self):
        raise error('group reference not allowed', pattern, self.position)

    def _compile(self, reverse, fuzzy):
        flags = 0
        if fuzzy:
            flags |= FUZZY_OP
        return [(self._opcode[self.case_flags, reverse], flags, self.group)]

    def dump(self, indent, reverse):
        print('{}REF_GROUP {}{}'.format(INDENT * indent, self.group, CASE_TEXT[self.case_flags]))

    def max_width(self):
        return UNLIMITED

    def __del__(self):
        self.info = None