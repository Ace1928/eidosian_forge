from tensorflow.python.autograph.pyct import qual_names
class Namer(object):
    """Symbol name generator."""

    def __init__(self, global_namespace):
        self.global_namespace = global_namespace
        self.generated_names = set()

    def new_symbol(self, name_root, reserved_locals):
        """See control_flow.SymbolNamer.new_symbol."""
        all_reserved_locals = set()
        for s in reserved_locals:
            if isinstance(s, qual_names.QN):
                all_reserved_locals.update(s.qn)
            elif isinstance(s, str):
                all_reserved_locals.add(s)
            else:
                raise ValueError('Unexpected symbol type "%s"' % type(s))
        pieces = name_root.split('_')
        if pieces[-1].isdigit():
            name_root = '_'.join(pieces[:-1])
            n = int(pieces[-1])
        else:
            n = 0
        new_name = name_root
        while new_name in self.global_namespace or new_name in all_reserved_locals or new_name in self.generated_names:
            n += 1
            new_name = '%s_%d' % (name_root, n)
        self.generated_names.add(new_name)
        return new_name