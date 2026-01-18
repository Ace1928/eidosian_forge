from numpy.testing import assert_
import numpy.distutils.fcompiler
class TestG77Versions:

    def test_g77_version(self):
        fc = numpy.distutils.fcompiler.new_fcompiler(compiler='gnu')
        for vs, version in g77_version_strings:
            v = fc.version_match(vs)
            assert_(v == version, (vs, v))

    def test_not_g77(self):
        fc = numpy.distutils.fcompiler.new_fcompiler(compiler='gnu')
        for vs, _ in gfortran_version_strings:
            v = fc.version_match(vs)
            assert_(v is None, (vs, v))