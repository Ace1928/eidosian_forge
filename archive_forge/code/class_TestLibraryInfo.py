import os
from numpy.distutils.npy_pkg_config import read_config, parse_flags
from numpy.testing import temppath, assert_
class TestLibraryInfo:

    def test_simple(self):
        with temppath('foo.ini') as path:
            with open(path, 'w') as f:
                f.write(simple)
            pkg = os.path.splitext(path)[0]
            out = read_config(pkg)
        assert_(out.cflags() == simple_d['cflags'])
        assert_(out.libs() == simple_d['libflags'])
        assert_(out.name == simple_d['name'])
        assert_(out.version == simple_d['version'])

    def test_simple_variable(self):
        with temppath('foo.ini') as path:
            with open(path, 'w') as f:
                f.write(simple_variable)
            pkg = os.path.splitext(path)[0]
            out = read_config(pkg)
        assert_(out.cflags() == simple_variable_d['cflags'])
        assert_(out.libs() == simple_variable_d['libflags'])
        assert_(out.name == simple_variable_d['name'])
        assert_(out.version == simple_variable_d['version'])
        out.vars['prefix'] = '/Users/david'
        assert_(out.cflags() == '-I/Users/david/include')