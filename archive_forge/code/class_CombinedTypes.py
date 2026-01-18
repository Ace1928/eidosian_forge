from inspect import isclass
class CombinedTypes(Type):
    """
            type resulting from the combination of other types

            """

    def __init__(self, *types):
        super(CombinedTypes, self).__init__(types=types)

    def iscombined(self):
        return True

    def generate(self, ctx):
        import sys
        current_recursion_limit = sys.getrecursionlimit()
        stypes = ordered_set()
        for t in self.types:
            try:
                stypes.append(ctx(t))
            except RecursionError:
                sys.setrecursionlimit(current_recursion_limit * 2)
                break
        if not stypes:
            sys.setrecursionlimit(current_recursion_limit)
            raise RecursionError
        elif len(stypes) == 1:
            sys.setrecursionlimit(current_recursion_limit)
            return stypes[0]
        else:
            stmp = 'typename __combined<{}>::type'.format(','.join(stypes))
            sys.setrecursionlimit(current_recursion_limit)
            return stmp