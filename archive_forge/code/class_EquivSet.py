import numpy
import operator
from numba.core import types, ir, config, cgutils, errors
from numba.core.ir_utils import (
from numba.core.analysis import compute_cfg_from_blocks
from numba.core.typing import npydecl, signature
import copy
from numba.core.extending import intrinsic
import llvmlite
class EquivSet(object):
    """EquivSet keeps track of equivalence relations between
    a set of objects.
    """

    def __init__(self, obj_to_ind=None, ind_to_obj=None, next_ind=0):
        """Create a new EquivSet object. Optional keyword arguments are for
        internal use only.
        """
        self.obj_to_ind = obj_to_ind if obj_to_ind else {}
        self.ind_to_obj = ind_to_obj if ind_to_obj else {}
        self.next_ind = next_ind

    def empty(self):
        """Return an empty EquivSet object.
        """
        return EquivSet()

    def clone(self):
        """Return a new copy.
        """
        return EquivSet(obj_to_ind=copy.deepcopy(self.obj_to_ind), ind_to_obj=copy.deepcopy(self.ind_to_obj), next_id=self.next_ind)

    def __repr__(self):
        return 'EquivSet({})'.format(self.ind_to_obj)

    def is_empty(self):
        """Return true if the set is empty, or false otherwise.
        """
        return self.obj_to_ind == {}

    def _get_ind(self, x):
        """Return the internal index (greater or equal to 0) of the given
        object, or -1 if not found.
        """
        return self.obj_to_ind.get(x, -1)

    def _get_or_add_ind(self, x):
        """Return the internal index (greater or equal to 0) of the given
        object, or create a new one if not found.
        """
        if x in self.obj_to_ind:
            i = self.obj_to_ind[x]
        else:
            i = self.next_ind
            self.next_ind += 1
        return i

    def _insert(self, objs):
        """Base method that inserts a set of equivalent objects by modifying
        self.
        """
        assert len(objs) > 1
        inds = tuple((self._get_or_add_ind(x) for x in objs))
        ind = min(inds)
        if config.DEBUG_ARRAY_OPT >= 2:
            print('_insert:', objs, inds)
        if not ind in self.ind_to_obj:
            self.ind_to_obj[ind] = []
        for i, obj in zip(inds, objs):
            if i == ind:
                if not obj in self.ind_to_obj[ind]:
                    self.ind_to_obj[ind].append(obj)
                    self.obj_to_ind[obj] = ind
            elif i in self.ind_to_obj:
                for x in self.ind_to_obj[i]:
                    self.obj_to_ind[x] = ind
                    self.ind_to_obj[ind].append(x)
                del self.ind_to_obj[i]
            else:
                self.obj_to_ind[obj] = ind
                self.ind_to_obj[ind].append(obj)

    def is_equiv(self, *objs):
        """Try to derive if given objects are equivalent, return true
        if so, or false otherwise.
        """
        inds = [self._get_ind(x) for x in objs]
        ind = max(inds)
        if ind != -1:
            return all((i == ind for i in inds))
        else:
            return all([x == objs[0] for x in objs])

    def get_equiv_const(self, obj):
        """Check if obj is equivalent to some int constant, and return
        the constant if found, or None otherwise.
        """
        ind = self._get_ind(obj)
        if ind >= 0:
            objs = self.ind_to_obj[ind]
            for x in objs:
                if isinstance(x, int):
                    return x
        return None

    def get_equiv_set(self, obj):
        """Return the set of equivalent objects.
        """
        ind = self._get_ind(obj)
        if ind >= 0:
            return set(self.ind_to_obj[ind])
        return set()

    def insert_equiv(self, *objs):
        """Insert a set of equivalent objects by modifying self. This
        method can be overloaded to transform object type before insertion.
        """
        return self._insert(objs)

    def intersect(self, equiv_set):
        """ Return the intersection of self and the given equiv_set,
        without modifying either of them. The result will also keep
        old equivalence indices unchanged.
        """
        new_set = self.empty()
        new_set.next_ind = self.next_ind
        for objs in equiv_set.ind_to_obj.values():
            inds = tuple((self._get_ind(x) for x in objs))
            ind_to_obj = {}
            for i, x in zip(inds, objs):
                if i in ind_to_obj:
                    ind_to_obj[i].append(x)
                elif i >= 0:
                    ind_to_obj[i] = [x]
            for v in ind_to_obj.values():
                if len(v) > 1:
                    new_set._insert(v)
        return new_set