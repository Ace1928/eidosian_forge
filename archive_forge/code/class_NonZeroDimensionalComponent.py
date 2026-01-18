from . import utilities
class NonZeroDimensionalComponent(Component):
    """
    Represents a non-zero dimensional component in the
    Ptolemy variety. It is a list that can hold points sampled from that
    component (witnesses).
    """

    def __init__(self, witnesses=[], dimension='unknown', free_variables=None, genus=None, p=None):
        if p is not None:
            self.dimension = p.dimension
            self.free_variables = p.free_variables
            self.genus = p.genus
        else:
            self.dimension = dimension
            self.free_variables = free_variables
            self.genus = genus
        super(NonZeroDimensionalComponent, self).__init__(witnesses)

    def _base_str_(self):
        if self.free_variables is None:
            f = ''
        else:
            f = ', free_variables = %r' % self.free_variables
        if self.genus is not None:
            f += ', genus = %d' % self.genus
        return 'NonZeroDimensionalComponent(dimension = %r%s)' % (self.dimension, f)

    def __repr__(self):
        base_str = self._base_str_()
        if len(self) > 0:
            l = ', '.join([repr(e) for e in self])
            return '[ %s (witnesses for %s) ]' % (l, base_str)
        return base_str

    def _repr_pretty_(self, p, cycle):
        base_str = self._base_str_()
        if cycle:
            p.text(base_str)
        elif len(self) > 0:
            with p.group(2, '[ ', ' ]'):
                for idx, item in enumerate(self):
                    if idx:
                        p.text(',')
                        p.breakable()
                    p.pretty(item)
                p.text(' ')
                p.breakable()
                p.text('(witnesses for %s)' % base_str)
        else:
            p.text(base_str)