import pytest
from numpy import array
from . import util
import platform
class TestReturnCharacter(util.F2PyTest):

    def check_function(self, t, tname):
        if tname in ['t0', 't1', 's0', 's1']:
            assert t('23') == b'2'
            r = t('ab')
            assert r == b'a'
            r = t(array('ab'))
            assert r == b'a'
            r = t(array(77, 'u1'))
            assert r == b'M'
        elif tname in ['ts', 'ss']:
            assert t(23) == b'23'
            assert t('123456789abcdef') == b'123456789a'
        elif tname in ['t5', 's5']:
            assert t(23) == b'23'
            assert t('ab') == b'ab'
            assert t('123456789abcdef') == b'12345'
        else:
            raise NotImplementedError