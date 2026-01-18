import pytest
from numpy import array
from . import util
class TestReturnInteger(util.F2PyTest):

    def check_function(self, t, tname):
        assert t(123) == 123
        assert t(123.6) == 123
        assert t('123') == 123
        assert t(-123) == -123
        assert t([123]) == 123
        assert t((123,)) == 123
        assert t(array(123)) == 123
        assert t(array(123, 'b')) == 123
        assert t(array(123, 'h')) == 123
        assert t(array(123, 'i')) == 123
        assert t(array(123, 'l')) == 123
        assert t(array(123, 'B')) == 123
        assert t(array(123, 'f')) == 123
        assert t(array(123, 'd')) == 123
        pytest.raises(ValueError, t, 'abc')
        pytest.raises(IndexError, t, [])
        pytest.raises(IndexError, t, ())
        pytest.raises(Exception, t, t)
        pytest.raises(Exception, t, {})
        if tname in ['t8', 's8']:
            pytest.raises(OverflowError, t, 100000000000000000000000)
            pytest.raises(OverflowError, t, 1.0000000011111112e+22)