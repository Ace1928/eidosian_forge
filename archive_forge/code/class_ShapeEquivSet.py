import numpy
import operator
from numba.core import types, ir, config, cgutils, errors
from numba.core.ir_utils import (
from numba.core.analysis import compute_cfg_from_blocks
from numba.core.typing import npydecl, signature
import copy
from numba.core.extending import intrinsic
import llvmlite
class ShapeEquivSet(EquivSet):
    """Just like EquivSet, except that it accepts only numba IR variables
    and constants as objects, guided by their types. Arrays are considered
    equivalent as long as their shapes are equivalent. Scalars are
    equivalent only when they are equal in value. Tuples are equivalent
    when they are of the same size, and their elements are equivalent.
    """

    def __init__(self, typemap, defs=None, ind_to_var=None, obj_to_ind=None, ind_to_obj=None, next_id=0, ind_to_const=None):
        """Create a new ShapeEquivSet object, where typemap is a dictionary
        that maps variable names to their types, and it will not be modified.
        Optional keyword arguments are for internal use only.
        """
        self.typemap = typemap
        self.defs = defs if defs else {}
        self.ind_to_var = ind_to_var if ind_to_var else {}
        self.ind_to_const = ind_to_const if ind_to_const else {}
        super(ShapeEquivSet, self).__init__(obj_to_ind, ind_to_obj, next_id)

    def empty(self):
        """Return an empty ShapeEquivSet.
        """
        return ShapeEquivSet(self.typemap, {})

    def clone(self):
        """Return a new copy.
        """
        return ShapeEquivSet(self.typemap, defs=copy.copy(self.defs), ind_to_var=copy.copy(self.ind_to_var), obj_to_ind=copy.deepcopy(self.obj_to_ind), ind_to_obj=copy.deepcopy(self.ind_to_obj), next_id=self.next_ind, ind_to_const=copy.deepcopy(self.ind_toconst))

    def __repr__(self):
        return 'ShapeEquivSet({}, ind_to_var={}, ind_to_const={})'.format(self.ind_to_obj, self.ind_to_var, self.ind_to_const)

    def _get_names(self, obj):
        """Return a set of names for the given obj, where array and tuples
        are broken down to their individual shapes or elements. This is
        safe because both Numba array shapes and Python tuples are immutable.
        """
        if isinstance(obj, ir.Var) or isinstance(obj, str):
            name = obj if isinstance(obj, str) else obj.name
            if name not in self.typemap:
                return (name,)
            typ = self.typemap[name]
            if isinstance(typ, (types.BaseTuple, types.ArrayCompatible)):
                ndim = typ.ndim if isinstance(typ, types.ArrayCompatible) else len(typ)
                if ndim == 0:
                    return (name,)
                else:
                    return tuple(('{}#{}'.format(name, i) for i in range(ndim)))
            else:
                return (name,)
        elif isinstance(obj, ir.Const):
            if isinstance(obj.value, tuple):
                return obj.value
            else:
                return (obj.value,)
        elif isinstance(obj, tuple):

            def get_names(x):
                names = self._get_names(x)
                if len(names) != 0:
                    return names[0]
                return names
            return tuple((get_names(x) for x in obj))
        elif isinstance(obj, int):
            return (obj,)
        if config.DEBUG_ARRAY_OPT >= 1:
            print(f'Ignoring untracked object type {type(obj)} in ShapeEquivSet')
        return ()

    def is_equiv(self, *objs):
        """Overload EquivSet.is_equiv to handle Numba IR variables and
        constants.
        """
        assert len(objs) > 1
        obj_names = [self._get_names(x) for x in objs]
        obj_names = [x for x in obj_names if x != ()]
        if len(obj_names) <= 1:
            return False
        ndims = [len(names) for names in obj_names]
        ndim = ndims[0]
        if not all((ndim == x for x in ndims)):
            if config.DEBUG_ARRAY_OPT >= 1:
                print('is_equiv: Dimension mismatch for {}'.format(objs))
            return False
        for i in range(ndim):
            names = [obj_name[i] for obj_name in obj_names]
            if not super(ShapeEquivSet, self).is_equiv(*names):
                return False
        return True

    def get_equiv_const(self, obj):
        """If the given object is equivalent to a constant scalar,
        return the scalar value, or None otherwise.
        """
        names = self._get_names(obj)
        if len(names) != 1:
            return None
        return super(ShapeEquivSet, self).get_equiv_const(names[0])

    def get_equiv_var(self, obj):
        """If the given object is equivalent to some defined variable,
        return the variable, or None otherwise.
        """
        names = self._get_names(obj)
        if len(names) != 1:
            return None
        ind = self._get_ind(names[0])
        vs = self.ind_to_var.get(ind, [])
        return vs[0] if vs != [] else None

    def get_equiv_set(self, obj):
        """Return the set of equivalent objects.
        """
        names = self._get_names(obj)
        if len(names) != 1:
            return None
        return super(ShapeEquivSet, self).get_equiv_set(names[0])

    def _insert(self, objs):
        """Overload EquivSet._insert to manage ind_to_var dictionary.
        """
        inds = []
        for obj in objs:
            if obj in self.obj_to_ind:
                inds.append(self.obj_to_ind[obj])
        varlist = []
        constval = None
        names = set()
        for i in sorted(inds):
            if i in self.ind_to_var:
                for x in self.ind_to_var[i]:
                    if not x.name in names:
                        varlist.append(x)
                        names.add(x.name)
            if i in self.ind_to_const:
                assert constval is None
                constval = self.ind_to_const[i]
        super(ShapeEquivSet, self)._insert(objs)
        new_ind = self.obj_to_ind[objs[0]]
        for i in set(inds):
            if i in self.ind_to_var:
                del self.ind_to_var[i]
        self.ind_to_var[new_ind] = varlist
        if constval is not None:
            self.ind_to_const[new_ind] = constval

    def insert_equiv(self, *objs):
        """Overload EquivSet.insert_equiv to handle Numba IR variables and
        constants. Input objs are either variable or constant, and at least
        one of them must be variable.
        """
        assert len(objs) > 1
        obj_names = [self._get_names(x) for x in objs]
        obj_names = [x for x in obj_names if x != ()]
        if len(obj_names) <= 1:
            return
        names = sum([list(x) for x in obj_names], [])
        ndims = [len(x) for x in obj_names]
        ndim = ndims[0]
        assert all((ndim == x for x in ndims)), 'Dimension mismatch for {}'.format(objs)
        varlist = []
        constlist = []
        for obj in objs:
            if not isinstance(obj, tuple):
                obj = (obj,)
            for var in obj:
                if isinstance(var, ir.Var) and (not var.name in varlist):
                    if var.name in self.defs:
                        varlist.insert(0, var)
                    else:
                        varlist.append(var)
                if isinstance(var, ir.Const) and (not var.value in constlist):
                    constlist.append(var.value)
        for obj in varlist:
            name = obj.name
            if name in names and (not name in self.obj_to_ind):
                self.ind_to_obj[self.next_ind] = [name]
                self.obj_to_ind[name] = self.next_ind
                self.ind_to_var[self.next_ind] = [obj]
                self.next_ind += 1
        for const in constlist:
            if const in names and (not const in self.obj_to_ind):
                self.ind_to_obj[self.next_ind] = [const]
                self.obj_to_ind[const] = self.next_ind
                self.ind_to_const[self.next_ind] = const
                self.next_ind += 1
        some_change = False
        for i in range(ndim):
            names = [obj_name[i] for obj_name in obj_names]
            ie_res = super(ShapeEquivSet, self).insert_equiv(*names)
            some_change = some_change or ie_res
        return some_change

    def has_shape(self, name):
        """Return true if the shape of the given variable is available.
        """
        return self.get_shape(name) is not None

    def get_shape(self, name):
        """Return a tuple of variables that corresponds to the shape
        of the given array, or None if not found.
        """
        return guard(self._get_shape, name)

    def _get_shape(self, name):
        """Return a tuple of variables that corresponds to the shape
        of the given array, or raise GuardException if not found.
        """
        inds = self.get_shape_classes(name)
        require(inds != ())
        shape = []
        for i in inds:
            require(i in self.ind_to_var)
            vs = self.ind_to_var[i]
            if vs != []:
                shape.append(vs[0])
            else:
                require(i in self.ind_to_const)
                vs = self.ind_to_const[i]
                shape.append(vs)
        return tuple(shape)

    def get_shape_classes(self, name):
        """Instead of the shape tuple, return tuple of int, where
        each int is the corresponding class index of the size object.
        Unknown shapes are given class index -1. Return empty tuple
        if the input name is a scalar variable.
        """
        if isinstance(name, ir.Var):
            name = name.name
        typ = self.typemap[name] if name in self.typemap else None
        if not isinstance(typ, (types.BaseTuple, types.SliceType, types.ArrayCompatible)):
            return []
        if isinstance(typ, types.ArrayCompatible) and typ.ndim == 0:
            return []
        names = self._get_names(name)
        inds = tuple((self._get_ind(name) for name in names))
        return inds

    def intersect(self, equiv_set):
        """Overload the intersect method to handle ind_to_var.
        """
        newset = super(ShapeEquivSet, self).intersect(equiv_set)
        ind_to_var = {}
        for i, objs in newset.ind_to_obj.items():
            assert len(objs) > 0
            obj = objs[0]
            assert obj in self.obj_to_ind
            assert obj in equiv_set.obj_to_ind
            j = self.obj_to_ind[obj]
            k = equiv_set.obj_to_ind[obj]
            assert j in self.ind_to_var
            assert k in equiv_set.ind_to_var
            varlist = []
            names = [x.name for x in equiv_set.ind_to_var[k]]
            for x in self.ind_to_var[j]:
                if x.name in names:
                    varlist.append(x)
            ind_to_var[i] = varlist
        newset.ind_to_var = ind_to_var
        return newset

    def define(self, name, redefined):
        """Increment the internal count of how many times a variable is being
        defined. Most variables in Numba IR are SSA, i.e., defined only once,
        but not all of them. When a variable is being re-defined, it must
        be removed from the equivalence relation and added to the redefined
        set but only if that redefinition is not known to have the same
        equivalence classes. Those variables redefined are removed from all
        the blocks' equivalence sets later.

        Arrays passed to define() use their whole name but these do not
        appear in the equivalence sets since they are stored there per
        dimension. Calling _get_names() here converts array names to
        dimensional names.

        This function would previously invalidate if there were any multiple
        definitions of a variable.  However, we realized that this behavior
        is overly restrictive.  You need only invalidate on multiple
        definitions if they are not known to be equivalent. So, the
        equivalence insertion functions now return True if some change was
        made (meaning the definition was not equivalent) and False
        otherwise. If no change was made, then define() need not be
        called. For no change to have been made, the variable must
        already be present. If the new definition of the var has the
        case where lhs and rhs are in the same equivalence class then
        again, no change will be made and define() need not be called
        or the variable invalidated.
        """
        if isinstance(name, ir.Var):
            name = name.name
        if name in self.defs:
            self.defs[name] += 1
            name_res = list(self._get_names(name))
            for one_name in name_res:
                if one_name in self.obj_to_ind:
                    redefined.add(one_name)
                    i = self.obj_to_ind[one_name]
                    del self.obj_to_ind[one_name]
                    self.ind_to_obj[i].remove(one_name)
                    if self.ind_to_obj[i] == []:
                        del self.ind_to_obj[i]
                    assert i in self.ind_to_var
                    names = [x.name for x in self.ind_to_var[i]]
                    if name in names:
                        j = names.index(name)
                        del self.ind_to_var[i][j]
                        if self.ind_to_var[i] == []:
                            del self.ind_to_var[i]
                            if i in self.ind_to_obj:
                                for obj in self.ind_to_obj[i]:
                                    del self.obj_to_ind[obj]
                                del self.ind_to_obj[i]
        else:
            self.defs[name] = 1

    def union_defs(self, defs, redefined):
        """Union with the given defs dictionary. This is meant to handle
        branch join-point, where a variable may have been defined in more
        than one branches.
        """
        for k, v in defs.items():
            if v > 0:
                self.define(k, redefined)