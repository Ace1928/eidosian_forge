from collections import deque
from numba.core import types, cgutils
class DataPacker(object):
    """
    A helper to pack a number of typed arguments into a data structure.
    Omitted arguments (i.e. values with the type `Omitted`) are automatically
    skipped.
    """

    def __init__(self, dmm, fe_types):
        self._dmm = dmm
        self._fe_types = fe_types
        self._models = [dmm.lookup(ty) for ty in fe_types]
        self._pack_map = []
        self._be_types = []
        for i, ty in enumerate(fe_types):
            if not isinstance(ty, types.Omitted):
                self._pack_map.append(i)
                self._be_types.append(self._models[i].get_data_type())

    def as_data(self, builder, values):
        """
        Return the given values packed as a data structure.
        """
        elems = [self._models[i].as_data(builder, values[i]) for i in self._pack_map]
        return cgutils.make_anonymous_struct(builder, elems)

    def _do_load(self, builder, ptr, formal_list=None):
        res = []
        for i, i_formal in enumerate(self._pack_map):
            elem_ptr = cgutils.gep_inbounds(builder, ptr, 0, i)
            val = self._models[i_formal].load_from_data_pointer(builder, elem_ptr)
            if formal_list is None:
                res.append((self._fe_types[i_formal], val))
            else:
                formal_list[i_formal] = val
        return res

    def load(self, builder, ptr):
        """
        Load the packed values and return a (type, value) tuples.
        """
        return self._do_load(builder, ptr)

    def load_into(self, builder, ptr, formal_list):
        """
        Load the packed values into a sequence indexed by formal
        argument number (skipping any Omitted position).
        """
        self._do_load(builder, ptr, formal_list)