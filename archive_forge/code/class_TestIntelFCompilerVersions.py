import numpy.distutils.fcompiler
from numpy.testing import assert_
class TestIntelFCompilerVersions:

    def test_32bit_version(self):
        fc = numpy.distutils.fcompiler.new_fcompiler(compiler='intel')
        for vs, version in intel_32bit_version_strings:
            v = fc.version_match(vs)
            assert_(v == version)