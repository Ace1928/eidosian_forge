from numpy.testing import assert_
import numpy.distutils.fcompiler
class TestGFortranVersions:

    def test_gfortran_version(self):
        fc = numpy.distutils.fcompiler.new_fcompiler(compiler='gnu95')
        for vs, version in gfortran_version_strings:
            v = fc.version_match(vs)
            assert_(v == version, (vs, v))

    def test_not_gfortran(self):
        fc = numpy.distutils.fcompiler.new_fcompiler(compiler='gnu95')
        for vs, _ in g77_version_strings:
            v = fc.version_match(vs)
            assert_(v is None, (vs, v))