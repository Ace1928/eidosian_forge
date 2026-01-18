from pathlib import Path
import re
import tokenize
from numpy.testing import assert_
import scipy
class TestFFTPackImport:

    def test_fftpack_import(self):
        base = Path(scipy.__file__).parent
        regexp = '\\s*from.+\\.fftpack import .*\\n'
        for path in base.rglob('*.py'):
            if base / 'fftpack' in path.parents:
                continue
            with tokenize.open(str(path)) as file:
                assert_(all((not re.fullmatch(regexp, line) for line in file)), f'{path} contains an import from fftpack')