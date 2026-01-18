from ._constants import *
class SubPattern:

    def __init__(self, state, data=None):
        self.state = state
        if data is None:
            data = []
        self.data = data
        self.width = None

    def dump(self, level=0):
        seqtypes = (tuple, list)
        for op, av in self.data:
            print(level * '  ' + str(op), end='')
            if op is IN:
                print()
                for op, a in av:
                    print((level + 1) * '  ' + str(op), a)
            elif op is BRANCH:
                print()
                for i, a in enumerate(av[1]):
                    if i:
                        print(level * '  ' + 'OR')
                    a.dump(level + 1)
            elif op is GROUPREF_EXISTS:
                condgroup, item_yes, item_no = av
                print('', condgroup)
                item_yes.dump(level + 1)
                if item_no:
                    print(level * '  ' + 'ELSE')
                    item_no.dump(level + 1)
            elif isinstance(av, SubPattern):
                print()
                av.dump(level + 1)
            elif isinstance(av, seqtypes):
                nl = False
                for a in av:
                    if isinstance(a, SubPattern):
                        if not nl:
                            print()
                        a.dump(level + 1)
                        nl = True
                    else:
                        if not nl:
                            print(' ', end='')
                        print(a, end='')
                        nl = False
                if not nl:
                    print()
            else:
                print('', av)

    def __repr__(self):
        return repr(self.data)

    def __len__(self):
        return len(self.data)

    def __delitem__(self, index):
        del self.data[index]

    def __getitem__(self, index):
        if isinstance(index, slice):
            return SubPattern(self.state, self.data[index])
        return self.data[index]

    def __setitem__(self, index, code):
        self.data[index] = code

    def insert(self, index, code):
        self.data.insert(index, code)

    def append(self, code):
        self.data.append(code)

    def getwidth(self):
        if self.width is not None:
            return self.width
        lo = hi = 0
        for op, av in self.data:
            if op is BRANCH:
                i = MAXWIDTH
                j = 0
                for av in av[1]:
                    l, h = av.getwidth()
                    i = min(i, l)
                    j = max(j, h)
                lo = lo + i
                hi = hi + j
            elif op is ATOMIC_GROUP:
                i, j = av.getwidth()
                lo = lo + i
                hi = hi + j
            elif op is SUBPATTERN:
                i, j = av[-1].getwidth()
                lo = lo + i
                hi = hi + j
            elif op in _REPEATCODES:
                i, j = av[2].getwidth()
                lo = lo + i * av[0]
                if av[1] == MAXREPEAT and j:
                    hi = MAXWIDTH
                else:
                    hi = hi + j * av[1]
            elif op in _UNITCODES:
                lo = lo + 1
                hi = hi + 1
            elif op is GROUPREF:
                i, j = self.state.groupwidths[av]
                lo = lo + i
                hi = hi + j
            elif op is GROUPREF_EXISTS:
                i, j = av[1].getwidth()
                if av[2] is not None:
                    l, h = av[2].getwidth()
                    i = min(i, l)
                    j = max(j, h)
                else:
                    i = 0
                lo = lo + i
                hi = hi + j
            elif op is SUCCESS:
                break
        self.width = (min(lo, MAXWIDTH), min(hi, MAXWIDTH))
        return self.width