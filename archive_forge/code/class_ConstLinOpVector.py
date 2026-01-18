from sys import version_info as _swig_python_version_info
class ConstLinOpVector(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def iterator(self):
        return _cvxcore.ConstLinOpVector_iterator(self)

    def __iter__(self):
        return self.iterator()

    def __nonzero__(self):
        return _cvxcore.ConstLinOpVector___nonzero__(self)

    def __bool__(self):
        return _cvxcore.ConstLinOpVector___bool__(self)

    def __len__(self):
        return _cvxcore.ConstLinOpVector___len__(self)

    def __getslice__(self, i, j):
        return _cvxcore.ConstLinOpVector___getslice__(self, i, j)

    def __setslice__(self, *args):
        return _cvxcore.ConstLinOpVector___setslice__(self, *args)

    def __delslice__(self, i, j):
        return _cvxcore.ConstLinOpVector___delslice__(self, i, j)

    def __delitem__(self, *args):
        return _cvxcore.ConstLinOpVector___delitem__(self, *args)

    def __getitem__(self, *args):
        return _cvxcore.ConstLinOpVector___getitem__(self, *args)

    def __setitem__(self, *args):
        return _cvxcore.ConstLinOpVector___setitem__(self, *args)

    def pop(self):
        return _cvxcore.ConstLinOpVector_pop(self)

    def append(self, x):
        return _cvxcore.ConstLinOpVector_append(self, x)

    def empty(self):
        return _cvxcore.ConstLinOpVector_empty(self)

    def size(self):
        return _cvxcore.ConstLinOpVector_size(self)

    def swap(self, v):
        return _cvxcore.ConstLinOpVector_swap(self, v)

    def begin(self):
        return _cvxcore.ConstLinOpVector_begin(self)

    def end(self):
        return _cvxcore.ConstLinOpVector_end(self)

    def rbegin(self):
        return _cvxcore.ConstLinOpVector_rbegin(self)

    def rend(self):
        return _cvxcore.ConstLinOpVector_rend(self)

    def clear(self):
        return _cvxcore.ConstLinOpVector_clear(self)

    def get_allocator(self):
        return _cvxcore.ConstLinOpVector_get_allocator(self)

    def pop_back(self):
        return _cvxcore.ConstLinOpVector_pop_back(self)

    def erase(self, *args):
        return _cvxcore.ConstLinOpVector_erase(self, *args)

    def __init__(self, *args):
        _cvxcore.ConstLinOpVector_swiginit(self, _cvxcore.new_ConstLinOpVector(*args))

    def push_back(self, x):
        return _cvxcore.ConstLinOpVector_push_back(self, x)

    def front(self):
        return _cvxcore.ConstLinOpVector_front(self)

    def back(self):
        return _cvxcore.ConstLinOpVector_back(self)

    def assign(self, n, x):
        return _cvxcore.ConstLinOpVector_assign(self, n, x)

    def resize(self, *args):
        return _cvxcore.ConstLinOpVector_resize(self, *args)

    def insert(self, *args):
        return _cvxcore.ConstLinOpVector_insert(self, *args)

    def reserve(self, n):
        return _cvxcore.ConstLinOpVector_reserve(self, n)

    def capacity(self):
        return _cvxcore.ConstLinOpVector_capacity(self)
    __swig_destroy__ = _cvxcore.delete_ConstLinOpVector