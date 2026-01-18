class InlineFragment(Selection):
    __slots__ = ('loc', 'type_condition', 'directives', 'selection_set')
    _fields = ('type_condition', 'directives', 'selection_set')

    def __init__(self, type_condition, selection_set, directives=None, loc=None):
        self.loc = loc
        self.type_condition = type_condition
        self.directives = directives
        self.selection_set = selection_set

    def __eq__(self, other):
        return self is other or (isinstance(other, InlineFragment) and self.type_condition == other.type_condition and (self.directives == other.directives) and (self.selection_set == other.selection_set))

    def __repr__(self):
        return 'InlineFragment(type_condition={self.type_condition!r}, directives={self.directives!r}, selection_set={self.selection_set!r})'.format(self=self)

    def __copy__(self):
        return type(self)(self.type_condition, self.selection_set, self.directives, self.loc)

    def __hash__(self):
        return id(self)