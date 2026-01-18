import pytest
import numpy as np
from numpy.testing import assert_warns
from numpy.ma.testutils import assert_equal
from numpy.ma.core import MaskedArrayFutureWarning
import io
import textwrap
class TestFromtextfile:

    def test_fromtextfile_delimitor(self):
        textfile = io.StringIO(textwrap.dedent("\n            A,B,C,D\n            'string 1';1;1.0;'mixed column'\n            'string 2';2;2.0;\n            'string 3';3;3.0;123\n            'string 4';4;4.0;3.14\n            "))
        with pytest.warns(DeprecationWarning):
            result = np.ma.mrecords.fromtextfile(textfile, delimitor=';')