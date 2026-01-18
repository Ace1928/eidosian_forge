from sys import version_info as _swig_python_version_info
class DoubleVector2D(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def iterator(self):
        return _cvxcore.DoubleVector2D_iterator(self)

    def __iter__(self):
        return self.iterator()

    def __nonzero__(self):
        return _cvxcore.DoubleVector2D___nonzero__(self)

    def __bool__(self):
        return _cvxcore.DoubleVector2D___bool__(self)

    def __len__(self):
        return _cvxcore.DoubleVector2D___len__(self)

    def __getslice__(self, i, j):
        return _cvxcore.DoubleVector2D___getslice__(self, i, j)

    def __setslice__(self, *args):
        return _cvxcore.DoubleVector2D___setslice__(self, *args)

    def __delslice__(self, i, j):
        return _cvxcore.DoubleVector2D___delslice__(self, i, j)

    def __delitem__(self, *args):
        return _cvxcore.DoubleVector2D___delitem__(self, *args)

    def __getitem__(self, *args):
        return _cvxcore.DoubleVector2D___getitem__(self, *args)

    def __setitem__(self, *args):
        return _cvxcore.DoubleVector2D___setitem__(self, *args)

    def pop(self):
        return _cvxcore.DoubleVector2D_pop(self)

    def append(self, x):
        return _cvxcore.DoubleVector2D_append(self, x)

    def empty(self):
        return _cvxcore.DoubleVector2D_empty(self)

    def size(self):
        return _cvxcore.DoubleVector2D_size(self)

    def swap(self, v):
        return _cvxcore.DoubleVector2D_swap(self, v)

    def begin(self):
        return _cvxcore.DoubleVector2D_begin(self)

    def end(self):
        return _cvxcore.DoubleVector2D_end(self)

    def rbegin(self):
        return _cvxcore.DoubleVector2D_rbegin(self)

    def rend(self):
        return _cvxcore.DoubleVector2D_rend(self)

    def clear(self):
        return _cvxcore.DoubleVector2D_clear(self)

    def get_allocator(self):
        return _cvxcore.DoubleVector2D_get_allocator(self)

    def pop_back(self):
        return _cvxcore.DoubleVector2D_pop_back(self)

    def erase(self, *args):
        return _cvxcore.DoubleVector2D_erase(self, *args)

    def __init__(self, *args):
        _cvxcore.DoubleVector2D_swiginit(self, _cvxcore.new_DoubleVector2D(*args))

    def push_back(self, x):
        return _cvxcore.DoubleVector2D_push_back(self, x)

    def front(self):
        return _cvxcore.DoubleVector2D_front(self)

    def back(self):
        return _cvxcore.DoubleVector2D_back(self)

    def assign(self, n, x):
        return _cvxcore.DoubleVector2D_assign(self, n, x)

    def resize(self, *args):
        return _cvxcore.DoubleVector2D_resize(self, *args)

    def insert(self, *args):
        return _cvxcore.DoubleVector2D_insert(self, *args)

    def reserve(self, n):
        return _cvxcore.DoubleVector2D_reserve(self, n)

    def capacity(self):
        return _cvxcore.DoubleVector2D_capacity(self)
    __swig_destroy__ = _cvxcore.delete_DoubleVector2D