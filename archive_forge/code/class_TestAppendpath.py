from os.path import join, sep, dirname
from numpy.distutils.misc_util import (
from numpy.testing import (
class TestAppendpath:

    def test_1(self):
        assert_equal(appendpath('prefix', 'name'), join('prefix', 'name'))
        assert_equal(appendpath('/prefix', 'name'), ajoin('prefix', 'name'))
        assert_equal(appendpath('/prefix', '/name'), ajoin('prefix', 'name'))
        assert_equal(appendpath('prefix', '/name'), join('prefix', 'name'))

    def test_2(self):
        assert_equal(appendpath('prefix/sub', 'name'), join('prefix', 'sub', 'name'))
        assert_equal(appendpath('prefix/sub', 'sup/name'), join('prefix', 'sub', 'sup', 'name'))
        assert_equal(appendpath('/prefix/sub', '/prefix/name'), ajoin('prefix', 'sub', 'name'))

    def test_3(self):
        assert_equal(appendpath('/prefix/sub', '/prefix/sup/name'), ajoin('prefix', 'sub', 'sup', 'name'))
        assert_equal(appendpath('/prefix/sub/sub2', '/prefix/sup/sup2/name'), ajoin('prefix', 'sub', 'sub2', 'sup', 'sup2', 'name'))
        assert_equal(appendpath('/prefix/sub/sub2', '/prefix/sub/sup/name'), ajoin('prefix', 'sub', 'sub2', 'sup', 'name'))