import time
from datetime import date
import numpy as np
from numpy.testing import (
from numpy.lib._iotools import (
class TestStringConverter:
    """Test StringConverter"""

    def test_creation(self):
        """Test creation of a StringConverter"""
        converter = StringConverter(int, -99999)
        assert_equal(converter._status, 1)
        assert_equal(converter.default, -99999)

    def test_upgrade(self):
        """Tests the upgrade method."""
        converter = StringConverter()
        assert_equal(converter._status, 0)
        assert_equal(converter.upgrade('0'), 0)
        assert_equal(converter._status, 1)
        import numpy.core.numeric as nx
        status_offset = int(nx.dtype(nx.int_).itemsize < nx.dtype(nx.int64).itemsize)
        assert_equal(converter.upgrade('17179869184'), 17179869184)
        assert_equal(converter._status, 1 + status_offset)
        assert_allclose(converter.upgrade('0.'), 0.0)
        assert_equal(converter._status, 2 + status_offset)
        assert_equal(converter.upgrade('0j'), complex('0j'))
        assert_equal(converter._status, 3 + status_offset)
        for s in ['a', b'a']:
            res = converter.upgrade(s)
            assert_(type(res) is str)
            assert_equal(res, 'a')
            assert_equal(converter._status, 8 + status_offset)

    def test_missing(self):
        """Tests the use of missing values."""
        converter = StringConverter(missing_values=('missing', 'missed'))
        converter.upgrade('0')
        assert_equal(converter('0'), 0)
        assert_equal(converter(''), converter.default)
        assert_equal(converter('missing'), converter.default)
        assert_equal(converter('missed'), converter.default)
        try:
            converter('miss')
        except ValueError:
            pass

    def test_upgrademapper(self):
        """Tests updatemapper"""
        dateparser = _bytes_to_date
        _original_mapper = StringConverter._mapper[:]
        try:
            StringConverter.upgrade_mapper(dateparser, date(2000, 1, 1))
            convert = StringConverter(dateparser, date(2000, 1, 1))
            test = convert('2001-01-01')
            assert_equal(test, date(2001, 1, 1))
            test = convert('2009-01-01')
            assert_equal(test, date(2009, 1, 1))
            test = convert('')
            assert_equal(test, date(2000, 1, 1))
        finally:
            StringConverter._mapper = _original_mapper

    def test_string_to_object(self):
        """Make sure that string-to-object functions are properly recognized"""
        old_mapper = StringConverter._mapper[:]
        conv = StringConverter(_bytes_to_date)
        assert_equal(conv._mapper, old_mapper)
        assert_(hasattr(conv, 'default'))

    def test_keep_default(self):
        """Make sure we don't lose an explicit default"""
        converter = StringConverter(None, missing_values='', default=-999)
        converter.upgrade('3.14159265')
        assert_equal(converter.default, -999)
        assert_equal(converter.type, np.dtype(float))
        converter = StringConverter(None, missing_values='', default=0)
        converter.upgrade('3.14159265')
        assert_equal(converter.default, 0)
        assert_equal(converter.type, np.dtype(float))

    def test_keep_default_zero(self):
        """Check that we don't lose a default of 0"""
        converter = StringConverter(int, default=0, missing_values='N/A')
        assert_equal(converter.default, 0)

    def test_keep_missing_values(self):
        """Check that we're not losing missing values"""
        converter = StringConverter(int, default=0, missing_values='N/A')
        assert_equal(converter.missing_values, {'', 'N/A'})

    def test_int64_dtype(self):
        """Check that int64 integer types can be specified"""
        converter = StringConverter(np.int64, default=0)
        val = '-9223372036854775807'
        assert_(converter(val) == -9223372036854775807)
        val = '9223372036854775807'
        assert_(converter(val) == 9223372036854775807)

    def test_uint64_dtype(self):
        """Check that uint64 integer types can be specified"""
        converter = StringConverter(np.uint64, default=0)
        val = '9223372043271415339'
        assert_(converter(val) == 9223372043271415339)