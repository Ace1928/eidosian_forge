import io
import os
import pytest
import sys
import rpy2.robjects as robjects
import rpy2.robjects.help
import rpy2.robjects.packages as packages
import rpy2.robjects.packages_utils
from rpy2.rinterface_lib.embedded import RRuntimeError
class TestPackage(object):

    def tests_package_from_env(self):
        env = robjects.Environment()
        env['a'] = robjects.StrVector('abcd')
        env['b'] = robjects.IntVector((1, 2, 3))
        env['c'] = robjects.r(' function(x) x^2')
        pck = robjects.packages.Package(env, 'dummy_package')
        assert isinstance(pck.a, robjects.Vector)
        assert isinstance(pck.b, robjects.Vector)
        assert isinstance(pck.c, robjects.Function)

    def test_new_with_dot(self):
        env = robjects.Environment()
        env['a.a'] = robjects.StrVector('abcd')
        env['b'] = robjects.IntVector((1, 2, 3))
        env['c'] = robjects.r(' function(x) x^2')
        pck = robjects.packages.Package(env, 'dummy_package')
        assert isinstance(pck.a_a, robjects.Vector)
        assert isinstance(pck.b, robjects.Vector)
        assert isinstance(pck.c, robjects.Function)

    def test_new_with_dot_conflict(self):
        env = robjects.Environment()
        env['a.a_a'] = robjects.StrVector('abcd')
        env['a_a.a'] = robjects.IntVector((1, 2, 3))
        env['c'] = robjects.r(' function(x) x^2')
        with pytest.raises(packages.LibraryError):
            robjects.packages.Package(env, 'dummy_package')

    def test_new_with_dot_conflict2(self):
        env = robjects.Environment()
        name_in_use = dir(packages.Package(env, 'foo'))[0]
        env[name_in_use] = robjects.StrVector('abcd')
        with pytest.raises(packages.LibraryError):
            robjects.packages.Package(env, 'dummy_package')

    def tests_package_repr(self):
        env = robjects.Environment()
        pck = robjects.packages.Package(env, 'dummy_package')
        assert isinstance(repr(pck), str)