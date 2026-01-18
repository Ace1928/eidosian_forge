import abc
import bz2
from datetime import date, datetime
from decimal import Decimal
import gc
import gzip
import io
import itertools
import os
import select
import shutil
import signal
import string
import tempfile
import threading
import time
import unittest
import weakref
import pytest
import numpy as np
import pyarrow as pa
from pyarrow.csv import (
from pyarrow.tests import util
class BaseCSVTableRead(BaseTestCSV):

    def read_csv(self, csv, *args, validate_full=True, **kwargs):
        """
        Reads the CSV file into memory using pyarrow's read_csv
        csv The CSV bytes
        args Positional arguments to be forwarded to pyarrow's read_csv
        validate_full Whether or not to fully validate the resulting table
        kwargs Keyword arguments to be forwarded to pyarrow's read_csv
        """
        assert isinstance(self.use_threads, bool)
        read_options = kwargs.setdefault('read_options', ReadOptions())
        read_options.use_threads = self.use_threads
        table = read_csv(csv, *args, **kwargs)
        table.validate(full=validate_full)
        return table

    def read_bytes(self, b, **kwargs):
        return self.read_csv(pa.py_buffer(b), **kwargs)

    def test_file_object(self):
        data = b'a,b\n1,2\n'
        expected_data = {'a': [1], 'b': [2]}
        bio = io.BytesIO(data)
        table = self.read_csv(bio)
        assert table.to_pydict() == expected_data
        sio = io.StringIO(data.decode())
        with pytest.raises(TypeError):
            self.read_csv(sio)

    def test_header(self):
        rows = b'abc,def,gh\n'
        table = self.read_bytes(rows)
        assert isinstance(table, pa.Table)
        self.check_names(table, ['abc', 'def', 'gh'])
        assert table.num_rows == 0

    def test_bom(self):
        rows = b'\xef\xbb\xbfa,b\n1,2\n'
        expected_data = {'a': [1], 'b': [2]}
        table = self.read_bytes(rows)
        assert table.to_pydict() == expected_data

    def test_one_chunk(self):
        rows = [b'a,b', b'1,2', b'3,4', b'56,78']
        for line_ending in [b'\n', b'\r', b'\r\n']:
            for file_ending in [b'', line_ending]:
                data = line_ending.join(rows) + file_ending
                table = self.read_bytes(data)
                assert len(table.to_batches()) == 1
                assert table.to_pydict() == {'a': [1, 3, 56], 'b': [2, 4, 78]}

    def test_header_column_names(self):
        rows = b'ab,cd\nef,gh\nij,kl\nmn,op\n'
        opts = ReadOptions()
        opts.column_names = ['x', 'y']
        table = self.read_bytes(rows, read_options=opts)
        self.check_names(table, ['x', 'y'])
        assert table.to_pydict() == {'x': ['ab', 'ef', 'ij', 'mn'], 'y': ['cd', 'gh', 'kl', 'op']}
        opts.skip_rows = 3
        table = self.read_bytes(rows, read_options=opts)
        self.check_names(table, ['x', 'y'])
        assert table.to_pydict() == {'x': ['mn'], 'y': ['op']}
        opts.skip_rows = 4
        table = self.read_bytes(rows, read_options=opts)
        self.check_names(table, ['x', 'y'])
        assert table.to_pydict() == {'x': [], 'y': []}
        opts.skip_rows = 5
        with pytest.raises(pa.ArrowInvalid):
            table = self.read_bytes(rows, read_options=opts)
        opts.skip_rows = 0
        opts.column_names = ['x', 'y', 'z']
        with pytest.raises(pa.ArrowInvalid, match='Expected 3 columns, got 2'):
            table = self.read_bytes(rows, read_options=opts)
        rows = b'abcd\n,,,,,\nij,kl\nmn,op\n'
        opts.skip_rows = 2
        opts.column_names = ['x', 'y']
        table = self.read_bytes(rows, read_options=opts)
        self.check_names(table, ['x', 'y'])
        assert table.to_pydict() == {'x': ['ij', 'mn'], 'y': ['kl', 'op']}

    def test_header_autogenerate_column_names(self):
        rows = b'ab,cd\nef,gh\nij,kl\nmn,op\n'
        opts = ReadOptions()
        opts.autogenerate_column_names = True
        table = self.read_bytes(rows, read_options=opts)
        self.check_names(table, ['f0', 'f1'])
        assert table.to_pydict() == {'f0': ['ab', 'ef', 'ij', 'mn'], 'f1': ['cd', 'gh', 'kl', 'op']}
        opts.skip_rows = 3
        table = self.read_bytes(rows, read_options=opts)
        self.check_names(table, ['f0', 'f1'])
        assert table.to_pydict() == {'f0': ['mn'], 'f1': ['op']}
        opts.skip_rows = 4
        with pytest.raises(pa.ArrowInvalid):
            table = self.read_bytes(rows, read_options=opts)

    def test_include_columns(self):
        rows = b'ab,cd\nef,gh\nij,kl\nmn,op\n'
        convert_options = ConvertOptions()
        convert_options.include_columns = ['ab']
        table = self.read_bytes(rows, convert_options=convert_options)
        self.check_names(table, ['ab'])
        assert table.to_pydict() == {'ab': ['ef', 'ij', 'mn']}
        convert_options.include_columns = ['cd', 'ab']
        table = self.read_bytes(rows, convert_options=convert_options)
        schema = pa.schema([('cd', pa.string()), ('ab', pa.string())])
        assert table.schema == schema
        assert table.to_pydict() == {'cd': ['gh', 'kl', 'op'], 'ab': ['ef', 'ij', 'mn']}
        convert_options.include_columns = ['xx', 'ab', 'yy']
        with pytest.raises(KeyError, match="Column 'xx' in include_columns does not exist in CSV file"):
            self.read_bytes(rows, convert_options=convert_options)

    def test_include_missing_columns(self):
        rows = b'ab,cd\nef,gh\nij,kl\nmn,op\n'
        read_options = ReadOptions()
        convert_options = ConvertOptions()
        convert_options.include_columns = ['xx', 'ab', 'yy']
        convert_options.include_missing_columns = True
        table = self.read_bytes(rows, read_options=read_options, convert_options=convert_options)
        schema = pa.schema([('xx', pa.null()), ('ab', pa.string()), ('yy', pa.null())])
        assert table.schema == schema
        assert table.to_pydict() == {'xx': [None, None, None], 'ab': ['ef', 'ij', 'mn'], 'yy': [None, None, None]}
        read_options.column_names = ['xx', 'yy']
        convert_options.include_columns = ['yy', 'cd']
        table = self.read_bytes(rows, read_options=read_options, convert_options=convert_options)
        schema = pa.schema([('yy', pa.string()), ('cd', pa.null())])
        assert table.schema == schema
        assert table.to_pydict() == {'yy': ['cd', 'gh', 'kl', 'op'], 'cd': [None, None, None, None]}
        convert_options.column_types = {'yy': pa.binary(), 'cd': pa.int32()}
        table = self.read_bytes(rows, read_options=read_options, convert_options=convert_options)
        schema = pa.schema([('yy', pa.binary()), ('cd', pa.int32())])
        assert table.schema == schema
        assert table.to_pydict() == {'yy': [b'cd', b'gh', b'kl', b'op'], 'cd': [None, None, None, None]}

    def test_simple_ints(self):
        rows = b'a,b,c\n1,2,3\n4,5,6\n'
        table = self.read_bytes(rows)
        schema = pa.schema([('a', pa.int64()), ('b', pa.int64()), ('c', pa.int64())])
        assert table.schema == schema
        assert table.to_pydict() == {'a': [1, 4], 'b': [2, 5], 'c': [3, 6]}

    def test_simple_varied(self):
        rows = b'a,b,c,d\n1,2,3,0\n4.0,-5,foo,True\n'
        table = self.read_bytes(rows)
        schema = pa.schema([('a', pa.float64()), ('b', pa.int64()), ('c', pa.string()), ('d', pa.bool_())])
        assert table.schema == schema
        assert table.to_pydict() == {'a': [1.0, 4.0], 'b': [2, -5], 'c': ['3', 'foo'], 'd': [False, True]}

    def test_simple_nulls(self):
        rows = b'a,b,c,d,e,f\n1,2,,,3,N/A\nnan,-5,foo,,nan,TRUE\n4.5,#N/A,nan,,\xff,false\n'
        table = self.read_bytes(rows)
        schema = pa.schema([('a', pa.float64()), ('b', pa.int64()), ('c', pa.string()), ('d', pa.null()), ('e', pa.binary()), ('f', pa.bool_())])
        assert table.schema == schema
        assert table.to_pydict() == {'a': [1.0, None, 4.5], 'b': [2, -5, None], 'c': ['', 'foo', 'nan'], 'd': [None, None, None], 'e': [b'3', b'nan', b'\xff'], 'f': [None, True, False]}

    def test_decimal_point(self):
        parse_options = ParseOptions(delimiter=';')
        rows = b'a;b\n1.25;2,5\nNA;-3\n-4;NA'
        table = self.read_bytes(rows, parse_options=parse_options)
        schema = pa.schema([('a', pa.float64()), ('b', pa.string())])
        assert table.schema == schema
        assert table.to_pydict() == {'a': [1.25, None, -4.0], 'b': ['2,5', '-3', 'NA']}
        convert_options = ConvertOptions(decimal_point=',')
        table = self.read_bytes(rows, parse_options=parse_options, convert_options=convert_options)
        schema = pa.schema([('a', pa.string()), ('b', pa.float64())])
        assert table.schema == schema
        assert table.to_pydict() == {'a': ['1.25', 'NA', '-4'], 'b': [2.5, -3.0, None]}

    def test_simple_timestamps(self):
        rows = b'a,b,c\n1970,1970-01-01 00:00:00,1970-01-01 00:00:00.123\n1989,1989-07-14 01:00:00,1989-07-14 01:00:00.123456\n'
        table = self.read_bytes(rows)
        schema = pa.schema([('a', pa.int64()), ('b', pa.timestamp('s')), ('c', pa.timestamp('ns'))])
        assert table.schema == schema
        assert table.to_pydict() == {'a': [1970, 1989], 'b': [datetime(1970, 1, 1), datetime(1989, 7, 14, 1)], 'c': [datetime(1970, 1, 1, 0, 0, 0, 123000), datetime(1989, 7, 14, 1, 0, 0, 123456)]}

    def test_timestamp_parsers(self):
        rows = b'a,b\n1970/01/01,1980-01-01 00\n1970/01/02,1980-01-02 00\n'
        opts = ConvertOptions()
        table = self.read_bytes(rows, convert_options=opts)
        schema = pa.schema([('a', pa.string()), ('b', pa.timestamp('s'))])
        assert table.schema == schema
        assert table.to_pydict() == {'a': ['1970/01/01', '1970/01/02'], 'b': [datetime(1980, 1, 1), datetime(1980, 1, 2)]}
        opts.timestamp_parsers = ['%Y/%m/%d']
        table = self.read_bytes(rows, convert_options=opts)
        schema = pa.schema([('a', pa.timestamp('s')), ('b', pa.string())])
        assert table.schema == schema
        assert table.to_pydict() == {'a': [datetime(1970, 1, 1), datetime(1970, 1, 2)], 'b': ['1980-01-01 00', '1980-01-02 00']}
        opts.timestamp_parsers = ['%Y/%m/%d', ISO8601]
        table = self.read_bytes(rows, convert_options=opts)
        schema = pa.schema([('a', pa.timestamp('s')), ('b', pa.timestamp('s'))])
        assert table.schema == schema
        assert table.to_pydict() == {'a': [datetime(1970, 1, 1), datetime(1970, 1, 2)], 'b': [datetime(1980, 1, 1), datetime(1980, 1, 2)]}

    def test_dates(self):
        rows = b'a,b\n1970-01-01,1970-01-02\n1971-01-01,1971-01-02\n'
        table = self.read_bytes(rows)
        schema = pa.schema([('a', pa.date32()), ('b', pa.date32())])
        assert table.schema == schema
        assert table.to_pydict() == {'a': [date(1970, 1, 1), date(1971, 1, 1)], 'b': [date(1970, 1, 2), date(1971, 1, 2)]}
        opts = ConvertOptions()
        opts.column_types = {'a': pa.date32(), 'b': pa.date64()}
        table = self.read_bytes(rows, convert_options=opts)
        schema = pa.schema([('a', pa.date32()), ('b', pa.date64())])
        assert table.schema == schema
        assert table.to_pydict() == {'a': [date(1970, 1, 1), date(1971, 1, 1)], 'b': [date(1970, 1, 2), date(1971, 1, 2)]}
        opts = ConvertOptions()
        opts.column_types = {'a': pa.timestamp('s'), 'b': pa.timestamp('ms')}
        table = self.read_bytes(rows, convert_options=opts)
        schema = pa.schema([('a', pa.timestamp('s')), ('b', pa.timestamp('ms'))])
        assert table.schema == schema
        assert table.to_pydict() == {'a': [datetime(1970, 1, 1), datetime(1971, 1, 1)], 'b': [datetime(1970, 1, 2), datetime(1971, 1, 2)]}

    def test_times(self):
        from datetime import time
        rows = b'a,b\n12:34:56,12:34:56.789\n23:59:59,23:59:59.999\n'
        table = self.read_bytes(rows)
        schema = pa.schema([('a', pa.time32('s')), ('b', pa.string())])
        assert table.schema == schema
        assert table.to_pydict() == {'a': [time(12, 34, 56), time(23, 59, 59)], 'b': ['12:34:56.789', '23:59:59.999']}
        opts = ConvertOptions()
        opts.column_types = {'a': pa.time64('us'), 'b': pa.time32('ms')}
        table = self.read_bytes(rows, convert_options=opts)
        schema = pa.schema([('a', pa.time64('us')), ('b', pa.time32('ms'))])
        assert table.schema == schema
        assert table.to_pydict() == {'a': [time(12, 34, 56), time(23, 59, 59)], 'b': [time(12, 34, 56, 789000), time(23, 59, 59, 999000)]}

    def test_auto_dict_encode(self):
        opts = ConvertOptions(auto_dict_encode=True)
        rows = 'a,b\nab,1\ncdé,2\ncdé,3\nab,4'.encode()
        table = self.read_bytes(rows, convert_options=opts)
        schema = pa.schema([('a', pa.dictionary(pa.int32(), pa.string())), ('b', pa.int64())])
        expected = {'a': ['ab', 'cdé', 'cdé', 'ab'], 'b': [1, 2, 3, 4]}
        assert table.schema == schema
        assert table.to_pydict() == expected
        opts.auto_dict_max_cardinality = 2
        table = self.read_bytes(rows, convert_options=opts)
        assert table.schema == schema
        assert table.to_pydict() == expected
        opts.auto_dict_max_cardinality = 1
        table = self.read_bytes(rows, convert_options=opts)
        assert table.schema == pa.schema([('a', pa.string()), ('b', pa.int64())])
        assert table.to_pydict() == expected
        opts.auto_dict_max_cardinality = 50
        opts.check_utf8 = False
        rows = b'a,b\nab,1\ncd\xff,2\nab,3'
        table = self.read_bytes(rows, convert_options=opts, validate_full=False)
        assert table.schema == schema
        dict_values = table['a'].chunk(0).dictionary
        assert len(dict_values) == 2
        assert dict_values[0].as_py() == 'ab'
        assert dict_values[1].as_buffer() == b'cd\xff'
        opts.check_utf8 = True
        table = self.read_bytes(rows, convert_options=opts)
        schema = pa.schema([('a', pa.dictionary(pa.int32(), pa.binary())), ('b', pa.int64())])
        expected = {'a': [b'ab', b'cd\xff', b'ab'], 'b': [1, 2, 3]}
        assert table.schema == schema
        assert table.to_pydict() == expected

    def test_custom_nulls(self):
        opts = ConvertOptions(null_values=['Xxx', 'Zzz'])
        rows = b'a,b,c,d\nZzz,"Xxx",1,2\nXxx,#N/A,,Zzz\n'
        table = self.read_bytes(rows, convert_options=opts)
        schema = pa.schema([('a', pa.null()), ('b', pa.string()), ('c', pa.string()), ('d', pa.int64())])
        assert table.schema == schema
        assert table.to_pydict() == {'a': [None, None], 'b': ['Xxx', '#N/A'], 'c': ['1', ''], 'd': [2, None]}
        opts = ConvertOptions(null_values=['Xxx', 'Zzz'], strings_can_be_null=True)
        table = self.read_bytes(rows, convert_options=opts)
        assert table.to_pydict() == {'a': [None, None], 'b': [None, '#N/A'], 'c': ['1', ''], 'd': [2, None]}
        opts.quoted_strings_can_be_null = False
        table = self.read_bytes(rows, convert_options=opts)
        assert table.to_pydict() == {'a': [None, None], 'b': ['Xxx', '#N/A'], 'c': ['1', ''], 'd': [2, None]}
        opts = ConvertOptions(null_values=[])
        rows = b'a,b\n#N/A,\n'
        table = self.read_bytes(rows, convert_options=opts)
        schema = pa.schema([('a', pa.string()), ('b', pa.string())])
        assert table.schema == schema
        assert table.to_pydict() == {'a': ['#N/A'], 'b': ['']}

    def test_custom_bools(self):
        opts = ConvertOptions(true_values=['T', 'yes'], false_values=['F', 'no'])
        rows = b'a,b,c\nTrue,T,t\nFalse,F,f\nTrue,yes,yes\nFalse,no,no\nN/A,N/A,N/A\n'
        table = self.read_bytes(rows, convert_options=opts)
        schema = pa.schema([('a', pa.string()), ('b', pa.bool_()), ('c', pa.string())])
        assert table.schema == schema
        assert table.to_pydict() == {'a': ['True', 'False', 'True', 'False', 'N/A'], 'b': [True, False, True, False, None], 'c': ['t', 'f', 'yes', 'no', 'N/A']}

    def test_column_types(self):
        opts = ConvertOptions(column_types={'b': 'float32', 'c': 'string', 'd': 'boolean', 'e': pa.decimal128(11, 2), 'zz': 'null'})
        rows = b'a,b,c,d,e\n1,2,3,true,1.0\n4,-5,6,false,0\n'
        table = self.read_bytes(rows, convert_options=opts)
        schema = pa.schema([('a', pa.int64()), ('b', pa.float32()), ('c', pa.string()), ('d', pa.bool_()), ('e', pa.decimal128(11, 2))])
        expected = {'a': [1, 4], 'b': [2.0, -5.0], 'c': ['3', '6'], 'd': [True, False], 'e': [Decimal('1.00'), Decimal('0.00')]}
        assert table.schema == schema
        assert table.to_pydict() == expected
        opts = ConvertOptions(column_types=pa.schema([('b', pa.float32()), ('c', pa.string()), ('d', pa.bool_()), ('e', pa.decimal128(11, 2)), ('zz', pa.bool_())]))
        table = self.read_bytes(rows, convert_options=opts)
        assert table.schema == schema
        assert table.to_pydict() == expected
        rows = b'a,b,c,d,e\n1,XXX,3,true,5\n4,-5,6,false,7\n'
        with pytest.raises(pa.ArrowInvalid) as exc:
            self.read_bytes(rows, convert_options=opts)
        err = str(exc.value)
        assert 'In CSV column #1: ' in err
        assert "CSV conversion error to float: invalid value 'XXX'" in err

    def test_column_types_dict(self):
        column_types = [('a', pa.dictionary(pa.int32(), pa.utf8())), ('b', pa.dictionary(pa.int32(), pa.int64())), ('c', pa.dictionary(pa.int32(), pa.decimal128(11, 2))), ('d', pa.dictionary(pa.int32(), pa.large_utf8()))]
        opts = ConvertOptions(column_types=dict(column_types))
        rows = b'a,b,c,d\nabc,123456,1.0,zz\ndefg,123456,0.5,xx\nabc,N/A,1.0,xx\n'
        table = self.read_bytes(rows, convert_options=opts)
        schema = pa.schema(column_types)
        expected = {'a': ['abc', 'defg', 'abc'], 'b': [123456, 123456, None], 'c': [Decimal('1.00'), Decimal('0.50'), Decimal('1.00')], 'd': ['zz', 'xx', 'xx']}
        assert table.schema == schema
        assert table.to_pydict() == expected
        column_types[0] = ('a', pa.dictionary(pa.int8(), pa.utf8()))
        opts = ConvertOptions(column_types=dict(column_types))
        with pytest.raises(NotImplementedError):
            table = self.read_bytes(rows, convert_options=opts)

    def test_column_types_with_column_names(self):
        rows = b'a,b\nc,d\ne,f\n'
        read_options = ReadOptions(column_names=['x', 'y'])
        convert_options = ConvertOptions(column_types={'x': pa.binary()})
        table = self.read_bytes(rows, read_options=read_options, convert_options=convert_options)
        schema = pa.schema([('x', pa.binary()), ('y', pa.string())])
        assert table.schema == schema
        assert table.to_pydict() == {'x': [b'a', b'c', b'e'], 'y': ['b', 'd', 'f']}

    def test_no_ending_newline(self):
        rows = b'a,b,c\n1,2,3\n4,5,6'
        table = self.read_bytes(rows)
        assert table.to_pydict() == {'a': [1, 4], 'b': [2, 5], 'c': [3, 6]}

    def test_trivial(self):
        rows = b',\n\n'
        table = self.read_bytes(rows)
        assert table.to_pydict() == {'': []}

    def test_empty_lines(self):
        rows = b'a,b\n\r1,2\r\n\r\n3,4\r\n'
        table = self.read_bytes(rows)
        assert table.to_pydict() == {'a': [1, 3], 'b': [2, 4]}
        parse_options = ParseOptions(ignore_empty_lines=False)
        table = self.read_bytes(rows, parse_options=parse_options)
        assert table.to_pydict() == {'a': [None, 1, None, 3], 'b': [None, 2, None, 4]}
        read_options = ReadOptions(skip_rows=2)
        table = self.read_bytes(rows, parse_options=parse_options, read_options=read_options)
        assert table.to_pydict() == {'1': [None, 3], '2': [None, 4]}

    def test_invalid_csv(self):
        rows = b'a,b,c\n1,2\n4,5,6\n'
        with pytest.raises(pa.ArrowInvalid, match='Expected 3 columns, got 2'):
            self.read_bytes(rows)
        rows = b'a,b,c\n1,2,3\n4'
        with pytest.raises(pa.ArrowInvalid, match='Expected 3 columns, got 1'):
            self.read_bytes(rows)
        for rows in [b'', b'\n', b'\r\n', b'\r', b'\n\n']:
            with pytest.raises(pa.ArrowInvalid, match='Empty CSV file'):
                self.read_bytes(rows)

    def test_options_delimiter(self):
        rows = b'a;b,c\nde,fg;eh\n'
        table = self.read_bytes(rows)
        assert table.to_pydict() == {'a;b': ['de'], 'c': ['fg;eh']}
        opts = ParseOptions(delimiter=';')
        table = self.read_bytes(rows, parse_options=opts)
        assert table.to_pydict() == {'a': ['de,fg'], 'b,c': ['eh']}

    def test_small_random_csv(self):
        csv, expected = make_random_csv(num_cols=2, num_rows=10)
        table = self.read_bytes(csv)
        assert table.schema == expected.schema
        assert table.equals(expected)
        assert table.to_pydict() == expected.to_pydict()

    def test_stress_block_sizes(self):
        csv_base, expected = make_random_csv(num_cols=2, num_rows=500)
        block_sizes = [11, 12, 13, 17, 37, 111]
        csvs = [csv_base, csv_base.rstrip(b'\r\n')]
        for csv in csvs:
            for block_size in block_sizes:
                read_options = ReadOptions(block_size=block_size)
                table = self.read_bytes(csv, read_options=read_options)
                assert table.schema == expected.schema
                if not table.equals(expected):
                    assert table.to_pydict() == expected.to_pydict()

    def test_stress_convert_options_blowup(self):
        try:
            clock = time.thread_time
        except AttributeError:
            clock = time.time
        num_columns = 10000
        col_names = ['K{}'.format(i) for i in range(num_columns)]
        csv = make_empty_csv(col_names)
        t1 = clock()
        convert_options = ConvertOptions(column_types={k: pa.string() for k in col_names[::2]})
        table = self.read_bytes(csv, convert_options=convert_options)
        dt = clock() - t1
        assert dt <= 10.0
        assert table.num_columns == num_columns
        assert table.num_rows == 0
        assert table.column_names == col_names

    def test_cancellation(self):
        if threading.current_thread().ident != threading.main_thread().ident:
            pytest.skip('test only works from main Python thread')
        raise_signal = util.get_raise_signal()
        signum = signal.SIGINT

        def signal_from_thread():
            time.sleep(0.2)
            raise_signal(signum)
        last_duration = 0.0
        workload_size = 100000
        attempts = 0
        while last_duration < 5.0 and attempts < 10:
            print('workload size:', workload_size)
            large_csv = b'a,b,c\n' + b'1,2,3\n' * workload_size
            exc_info = None
            try:
                with util.signal_wakeup_fd() as sigfd:
                    try:
                        t = threading.Thread(target=signal_from_thread)
                        t.start()
                        t1 = time.time()
                        try:
                            self.read_bytes(large_csv)
                        except KeyboardInterrupt as e:
                            exc_info = e
                            last_duration = time.time() - t1
                    finally:
                        select.select([sigfd], [], [sigfd], 10.0)
            except KeyboardInterrupt:
                pass
            if exc_info is not None:
                if exc_info.__context__ is not None:
                    break
            workload_size = workload_size * 3
        if exc_info is None:
            pytest.fail('Failed to get an interruption during CSV reading')
        assert last_duration <= 1.0
        e = exc_info.__context__
        assert isinstance(e, pa.ArrowCancelled)
        assert e.signum == signum

    def test_cancellation_disabled(self):
        t = threading.Thread(target=lambda: self.read_bytes(b'f64\n0.1'))
        t.start()
        t.join()