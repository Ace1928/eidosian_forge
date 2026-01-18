import enum
import string
import unicodedata
from collections import defaultdict
import regex._regex as _regex
class LookAroundConditional(RegexBase):
    _dir_text = {False: 'AHEAD', True: 'BEHIND'}

    def __init__(self, behind, positive, subpattern, yes_item, no_item):
        RegexBase.__init__(self)
        self.behind = bool(behind)
        self.positive = bool(positive)
        self.subpattern = subpattern
        self.yes_item = yes_item
        self.no_item = no_item

    def fix_groups(self, pattern, reverse, fuzzy):
        self.subpattern.fix_groups(pattern, reverse, fuzzy)
        self.yes_item.fix_groups(pattern, reverse, fuzzy)
        self.no_item.fix_groups(pattern, reverse, fuzzy)

    def optimise(self, info, reverse):
        subpattern = self.subpattern.optimise(info, self.behind)
        yes_item = self.yes_item.optimise(info, self.behind)
        no_item = self.no_item.optimise(info, self.behind)
        return LookAroundConditional(self.behind, self.positive, subpattern, yes_item, no_item)

    def pack_characters(self, info):
        self.subpattern = self.subpattern.pack_characters(info)
        self.yes_item = self.yes_item.pack_characters(info)
        self.no_item = self.no_item.pack_characters(info)
        return self

    def remove_captures(self):
        self.subpattern = self.subpattern.remove_captures()
        self.yes_item = self.yes_item.remove_captures()
        self.no_item = self.no_item.remove_captures()

    def is_atomic(self):
        return self.subpattern.is_atomic() and self.yes_item.is_atomic() and self.no_item.is_atomic()

    def can_be_affix(self):
        return self.subpattern.can_be_affix() and self.yes_item.can_be_affix() and self.no_item.can_be_affix()

    def contains_group(self):
        return self.subpattern.contains_group() or self.yes_item.contains_group() or self.no_item.contains_group()

    def _compile(self, reverse, fuzzy):
        code = [(OP.CONDITIONAL, int(self.positive), int(not self.behind))]
        code.extend(self.subpattern.compile(self.behind, fuzzy))
        code.append((OP.NEXT,))
        code.extend(self.yes_item.compile(reverse, fuzzy))
        add_code = self.no_item.compile(reverse, fuzzy)
        if add_code:
            code.append((OP.NEXT,))
            code.extend(add_code)
        code.append((OP.END,))
        return code

    def dump(self, indent, reverse):
        print('{}CONDITIONAL {} {}'.format(INDENT * indent, self._dir_text[self.behind], POS_TEXT[self.positive]))
        self.subpattern.dump(indent + 1, self.behind)
        print('{}EITHER'.format(INDENT * indent))
        self.yes_item.dump(indent + 1, reverse)
        if not self.no_item.is_empty():
            print('{}OR'.format(INDENT * indent))
            self.no_item.dump(indent + 1, reverse)

    def is_empty(self):
        return self.subpattern.is_empty() and self.yes_item.is_empty() or self.no_item.is_empty()

    def __eq__(self, other):
        return type(self) is type(other) and (self.subpattern, self.yes_item, self.no_item) == (other.subpattern, other.yes_item, other.no_item)

    def max_width(self):
        return max(self.yes_item.max_width(), self.no_item.max_width())

    def get_required_string(self, reverse):
        return (self.max_width(), None)