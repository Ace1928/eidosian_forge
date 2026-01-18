from __future__ import absolute_import
import copy
import hashlib
import re
from functools import partial
from itertools import product
from Cython.Utils import cached_function
from .Code import UtilityCode, LazyUtilityCode, TempitaUtilityCode
from . import StringEncoding
from . import Naming
from .Errors import error, CannotSpecialize, performance_hint
class MemoryViewSliceType(PyrexType):
    is_memoryviewslice = 1
    default_value = '{ 0, 0, { 0 }, { 0 }, { 0 } }'
    has_attributes = 1
    needs_refcounting = 1
    refcounting_needs_gil = False
    scope = None
    from_py_function = None
    to_py_function = None
    exception_value = None
    exception_check = True
    subtypes = ['dtype']

    def __init__(self, base_dtype, axes):
        """
        MemoryViewSliceType(base, axes)

        Base is the C base type; axes is a list of (access, packing) strings,
        where access is one of 'full', 'direct' or 'ptr' and packing is one of
        'contig', 'strided' or 'follow'.  There is one (access, packing) tuple
        for each dimension.

        the access specifiers determine whether the array data contains
        pointers that need to be dereferenced along that axis when
        retrieving/setting:

        'direct' -- No pointers stored in this dimension.
        'ptr' -- Pointer stored in this dimension.
        'full' -- Check along this dimension, don't assume either.

        the packing specifiers specify how the array elements are laid-out
        in memory.

        'contig' -- The data is contiguous in memory along this dimension.
                At most one dimension may be specified as 'contig'.
        'strided' -- The data isn't contiguous along this dimension.
        'follow' -- Used for C/Fortran contiguous arrays, a 'follow' dimension
            has its stride automatically computed from extents of the other
            dimensions to ensure C or Fortran memory layout.

        C-contiguous memory has 'direct' as the access spec, 'contig' as the
        *last* axis' packing spec and 'follow' for all other packing specs.

        Fortran-contiguous memory has 'direct' as the access spec, 'contig' as
        the *first* axis' packing spec and 'follow' for all other packing
        specs.
        """
        from . import Buffer, MemoryView
        self.dtype = base_dtype
        self.axes = axes
        self.ndim = len(axes)
        self.flags = MemoryView.get_buf_flags(self.axes)
        self.is_c_contig, self.is_f_contig = MemoryView.is_cf_contig(self.axes)
        assert not (self.is_c_contig and self.is_f_contig)
        self.mode = MemoryView.get_mode(axes)
        self.writable_needed = False
        if not self.dtype.is_fused:
            self.dtype_name = Buffer.mangle_dtype_name(self.dtype)

    def __hash__(self):
        return hash(self.__class__) ^ hash(self.dtype) ^ hash(tuple(self.axes))

    def __eq__(self, other):
        if isinstance(other, BaseType):
            return self.same_as_resolved_type(other)
        else:
            return False

    def __ne__(self, other):
        return not self == other

    def same_as_resolved_type(self, other_type):
        return other_type.is_memoryviewslice and self.dtype.same_as(other_type.dtype) and (self.axes == other_type.axes) or other_type is error_type

    def needs_nonecheck(self):
        return True

    def is_complete(self):
        return 0

    def can_be_optional(self):
        """Returns True if type can be used with typing.Optional[]."""
        return True

    def declaration_code(self, entity_code, for_display=0, dll_linkage=None, pyrex=0):
        assert not dll_linkage
        from . import MemoryView
        base_code = StringEncoding.EncodedString(str(self) if pyrex or for_display else MemoryView.memviewslice_cname)
        return self.base_declaration_code(base_code, entity_code)

    def attributes_known(self):
        if self.scope is None:
            from . import Symtab
            self.scope = scope = Symtab.CClassScope('mvs_class_' + self.specialization_suffix(), None, visibility='extern', parent_type=self)
            scope.directives = {}
            scope.declare_var('_data', c_char_ptr_type, None, cname='data', is_cdef=1)
        return True

    def declare_attribute(self, attribute, env, pos):
        from . import MemoryView, Options
        scope = self.scope
        if attribute == 'shape':
            scope.declare_var('shape', c_array_type(c_py_ssize_t_type, Options.buffer_max_dims), pos, cname='shape', is_cdef=1)
        elif attribute == 'strides':
            scope.declare_var('strides', c_array_type(c_py_ssize_t_type, Options.buffer_max_dims), pos, cname='strides', is_cdef=1)
        elif attribute == 'suboffsets':
            scope.declare_var('suboffsets', c_array_type(c_py_ssize_t_type, Options.buffer_max_dims), pos, cname='suboffsets', is_cdef=1)
        elif attribute in ('copy', 'copy_fortran'):
            ndim = len(self.axes)
            follow_dim = [('direct', 'follow')]
            contig_dim = [('direct', 'contig')]
            to_axes_c = follow_dim * (ndim - 1) + contig_dim
            to_axes_f = contig_dim + follow_dim * (ndim - 1)
            dtype = self.dtype
            if dtype.is_cv_qualified:
                dtype = dtype.cv_base_type
            to_memview_c = MemoryViewSliceType(dtype, to_axes_c)
            to_memview_f = MemoryViewSliceType(dtype, to_axes_f)
            for to_memview, cython_name in [(to_memview_c, 'copy'), (to_memview_f, 'copy_fortran')]:
                copy_func_type = CFuncType(to_memview, [CFuncTypeArg('memviewslice', self, None)])
                copy_cname = MemoryView.copy_c_or_fortran_cname(to_memview)
                entry = scope.declare_cfunction(cython_name, copy_func_type, pos=pos, defining=1, cname=copy_cname)
                utility = MemoryView.get_copy_new_utility(pos, self, to_memview)
                env.use_utility_code(utility)
            MemoryView.use_cython_array_utility_code(env)
        elif attribute in ('is_c_contig', 'is_f_contig'):
            for c_or_f, cython_name in (('C', 'is_c_contig'), ('F', 'is_f_contig')):
                is_contig_name = MemoryView.get_is_contig_func_name(c_or_f, self.ndim)
                cfunctype = CFuncType(return_type=c_bint_type, args=[CFuncTypeArg('memviewslice', self, None)], exception_value='-1')
                entry = scope.declare_cfunction(cython_name, cfunctype, pos=pos, defining=1, cname=is_contig_name)
                entry.utility_code_definition = MemoryView.get_is_contig_utility(c_or_f, self.ndim)
        return True

    def get_entry(self, node, cname=None, type=None):
        from . import MemoryView, Symtab
        if cname is None:
            assert node.is_simple() or node.is_temp or node.is_elemental
            cname = node.result()
        if type is None:
            type = node.type
        entry = Symtab.Entry(cname, cname, type, node.pos)
        return MemoryView.MemoryViewSliceBufferEntry(entry)

    def conforms_to(self, dst, broadcast=False, copying=False):
        """
        Returns True if src conforms to dst, False otherwise.

        If conformable, the types are the same, the ndims are equal, and each axis spec is conformable.

        Any packing/access spec is conformable to itself.

        'direct' and 'ptr' are conformable to 'full'.
        'contig' and 'follow' are conformable to 'strided'.
        Any other combo is not conformable.
        """
        from . import MemoryView
        src = self
        src_dtype, dst_dtype = (src.dtype, dst.dtype)
        if not copying:
            if src_dtype.is_const and (not dst_dtype.is_const):
                return False
            if src_dtype.is_volatile and (not dst_dtype.is_volatile):
                return False
        if src_dtype.is_cv_qualified:
            src_dtype = src_dtype.cv_base_type
        if dst_dtype.is_cv_qualified:
            dst_dtype = dst_dtype.cv_base_type
        if not src_dtype.same_as(dst_dtype):
            return False
        if src.ndim != dst.ndim:
            if broadcast:
                src, dst = MemoryView.broadcast_types(src, dst)
            else:
                return False
        for src_spec, dst_spec in zip(src.axes, dst.axes):
            src_access, src_packing = src_spec
            dst_access, dst_packing = dst_spec
            if src_access != dst_access and dst_access != 'full':
                return False
            if src_packing != dst_packing and dst_packing != 'strided' and (not copying):
                return False
        return True

    def valid_dtype(self, dtype, i=0):
        """
        Return whether type dtype can be used as the base type of a
        memoryview slice.

        We support structs, numeric types and objects
        """
        if dtype.is_complex and dtype.real_type.is_int:
            return False
        if dtype.is_struct and dtype.kind == 'struct':
            for member in dtype.scope.var_entries:
                if not self.valid_dtype(member.type):
                    return False
            return True
        return dtype.is_error or (dtype.is_array and i < 8 and self.valid_dtype(dtype.base_type, i + 1)) or dtype.is_numeric or dtype.is_pyobject or dtype.is_fused or (dtype.is_typedef and self.valid_dtype(dtype.typedef_base_type))

    def validate_memslice_dtype(self, pos):
        if not self.valid_dtype(self.dtype):
            error(pos, 'Invalid base type for memoryview slice: %s' % self.dtype)

    def assert_direct_dims(self, pos):
        for access, packing in self.axes:
            if access != 'direct':
                error(pos, 'All dimensions must be direct')
                return False
        return True

    def transpose(self, pos):
        if not self.assert_direct_dims(pos):
            return error_type
        return MemoryViewSliceType(self.dtype, self.axes[::-1])

    def specialization_name(self):
        return '%s_%s' % (super(MemoryViewSliceType, self).specialization_name(), self.specialization_suffix())

    def specialization_suffix(self):
        return '%s_%s' % (self.axes_to_name(), self.dtype_name)

    def can_coerce_to_pyobject(self, env):
        return True

    def can_coerce_from_pyobject(self, env):
        return True

    def check_for_null_code(self, cname):
        return cname + '.memview'

    def create_from_py_utility_code(self, env):
        from . import MemoryView, Buffer

        def lazy_utility_callback(code):
            context['dtype_typeinfo'] = Buffer.get_type_information_cname(code, self.dtype)
            return TempitaUtilityCode.load('ObjectToMemviewSlice', 'MemoryView_C.c', context=context)
        env.use_utility_code(MemoryView.memviewslice_init_code)
        env.use_utility_code(LazyUtilityCode(lazy_utility_callback))
        if self.is_c_contig:
            c_or_f_flag = '__Pyx_IS_C_CONTIG'
        elif self.is_f_contig:
            c_or_f_flag = '__Pyx_IS_F_CONTIG'
        else:
            c_or_f_flag = '0'
        suffix = self.specialization_suffix()
        funcname = '__Pyx_PyObject_to_MemoryviewSlice_' + suffix
        context = dict(MemoryView.context, buf_flag=self.flags, ndim=self.ndim, axes_specs=', '.join(self.axes_to_code()), dtype_typedecl=self.dtype.empty_declaration_code(), struct_nesting_depth=self.dtype.struct_nesting_depth(), c_or_f_flag=c_or_f_flag, funcname=funcname)
        self.from_py_function = funcname
        return True

    def from_py_call_code(self, source_code, result_code, error_pos, code, from_py_function=None, error_condition=None, special_none_cvalue=None):
        writable = not self.dtype.is_const
        return self._assign_from_py_code(source_code, result_code, error_pos, code, from_py_function, error_condition, extra_args=['PyBUF_WRITABLE' if writable else '0'], special_none_cvalue=special_none_cvalue)

    def create_to_py_utility_code(self, env):
        self._dtype_to_py_func, self._dtype_from_py_func = self.dtype_object_conversion_funcs(env)
        return True

    def to_py_call_code(self, source_code, result_code, result_type, to_py_function=None):
        assert self._dtype_to_py_func
        assert self._dtype_from_py_func
        to_py_func = '(PyObject *(*)(char *)) ' + self._dtype_to_py_func
        from_py_func = '(int (*)(char *, PyObject *)) ' + self._dtype_from_py_func
        tup = (result_code, source_code, self.ndim, to_py_func, from_py_func, self.dtype.is_pyobject)
        return '%s = __pyx_memoryview_fromslice(%s, %s, %s, %s, %d);' % tup

    def dtype_object_conversion_funcs(self, env):
        get_function = '__pyx_memview_get_%s' % self.dtype_name
        set_function = '__pyx_memview_set_%s' % self.dtype_name
        context = dict(get_function=get_function, set_function=set_function)
        if self.dtype.is_pyobject:
            utility_name = 'MemviewObjectToObject'
        else:
            self.dtype.create_to_py_utility_code(env)
            to_py_function = self.dtype.to_py_function
            from_py_function = None
            if not self.dtype.is_const:
                self.dtype.create_from_py_utility_code(env)
                from_py_function = self.dtype.from_py_function
            if not (to_py_function or from_py_function):
                return ('NULL', 'NULL')
            if not to_py_function:
                get_function = 'NULL'
            if not from_py_function:
                set_function = 'NULL'
            utility_name = 'MemviewDtypeToObject'
            error_condition = self.dtype.error_condition('value') or 'PyErr_Occurred()'
            context.update(to_py_function=to_py_function, from_py_function=from_py_function, dtype=self.dtype.empty_declaration_code(), error_condition=error_condition)
        utility = TempitaUtilityCode.load_cached(utility_name, 'MemoryView_C.c', context=context)
        env.use_utility_code(utility)
        return (get_function, set_function)

    def axes_to_code(self):
        """Return a list of code constants for each axis"""
        from . import MemoryView
        d = MemoryView._spec_to_const
        return ['(%s | %s)' % (d[a], d[p]) for a, p in self.axes]

    def axes_to_name(self):
        """Return an abbreviated name for our axes"""
        from . import MemoryView
        d = MemoryView._spec_to_abbrev
        return ''.join(['%s%s' % (d[a], d[p]) for a, p in self.axes])

    def error_condition(self, result_code):
        return '!%s.memview' % result_code

    def __str__(self):
        from . import MemoryView
        axes_code_list = []
        for idx, (access, packing) in enumerate(self.axes):
            flag = MemoryView.get_memoryview_flag(access, packing)
            if flag == 'strided':
                axes_code_list.append(':')
            else:
                if flag == 'contiguous':
                    have_follow = [p for a, p in self.axes[idx - 1:idx + 2] if p == 'follow']
                    if have_follow or self.ndim == 1:
                        flag = '1'
                axes_code_list.append('::' + flag)
        if self.dtype.is_pyobject:
            dtype_name = self.dtype.name
        else:
            dtype_name = self.dtype
        return '%s[%s]' % (dtype_name, ', '.join(axes_code_list))

    def specialize(self, values):
        """This does not validate the base type!!"""
        dtype = self.dtype.specialize(values)
        if dtype is not self.dtype:
            return MemoryViewSliceType(dtype, self.axes)
        return self

    def cast_code(self, expr_code):
        return expr_code

    def generate_incref(self, code, name, **kwds):
        pass

    def generate_incref_memoryviewslice(self, code, slice_cname, have_gil):
        code.putln('__PYX_INC_MEMVIEW(&%s, %d);' % (slice_cname, int(have_gil)))

    def generate_xdecref(self, code, cname, nanny, have_gil):
        code.putln('__PYX_XCLEAR_MEMVIEW(&%s, %d);' % (cname, int(have_gil)))

    def generate_decref(self, code, cname, nanny, have_gil):
        self.generate_xdecref(code, cname, nanny, have_gil)

    def generate_xdecref_clear(self, code, cname, clear_before_decref, **kwds):
        self.generate_xdecref(code, cname, **kwds)
        code.putln('%s.memview = NULL; %s.data = NULL;' % (cname, cname))

    def generate_decref_clear(self, code, cname, **kwds):
        self.generate_xdecref_clear(code, cname, **kwds)
    generate_gotref = generate_xgotref = generate_xgiveref = generate_giveref = lambda *args: None