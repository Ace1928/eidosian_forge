import json
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.io.json._normalize import nested_to_record
class TestJSONNormalize:

    def test_simple_records(self):
        recs = [{'a': 1, 'b': 2, 'c': 3}, {'a': 4, 'b': 5, 'c': 6}, {'a': 7, 'b': 8, 'c': 9}, {'a': 10, 'b': 11, 'c': 12}]
        result = json_normalize(recs)
        expected = DataFrame(recs)
        tm.assert_frame_equal(result, expected)

    def test_simple_normalize(self, state_data):
        result = json_normalize(state_data[0], 'counties')
        expected = DataFrame(state_data[0]['counties'])
        tm.assert_frame_equal(result, expected)
        result = json_normalize(state_data, 'counties')
        expected = []
        for rec in state_data:
            expected.extend(rec['counties'])
        expected = DataFrame(expected)
        tm.assert_frame_equal(result, expected)
        result = json_normalize(state_data, 'counties', meta='state')
        expected['state'] = np.array(['Florida', 'Ohio']).repeat([3, 2])
        tm.assert_frame_equal(result, expected)

    def test_fields_list_type_normalize(self):
        parse_metadata_fields_list_type = [{'values': [1, 2, 3], 'metadata': {'listdata': [1, 2]}}]
        result = json_normalize(parse_metadata_fields_list_type, record_path=['values'], meta=[['metadata', 'listdata']])
        expected = DataFrame({0: [1, 2, 3], 'metadata.listdata': [[1, 2], [1, 2], [1, 2]]})
        tm.assert_frame_equal(result, expected)

    def test_empty_array(self):
        result = json_normalize([])
        expected = DataFrame()
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('data, record_path, exception_type', [([{'a': 0}, {'a': 1}], None, None), ({'a': [{'a': 0}, {'a': 1}]}, 'a', None), ('{"a": [{"a": 0}, {"a": 1}]}', None, NotImplementedError), (None, None, NotImplementedError)])
    def test_accepted_input(self, data, record_path, exception_type):
        if exception_type is not None:
            with pytest.raises(exception_type, match=''):
                json_normalize(data, record_path=record_path)
        else:
            result = json_normalize(data, record_path=record_path)
            expected = DataFrame([0, 1], columns=['a'])
            tm.assert_frame_equal(result, expected)

    def test_simple_normalize_with_separator(self, deep_nested):
        result = json_normalize({'A': {'A': 1, 'B': 2}})
        expected = DataFrame([[1, 2]], columns=['A.A', 'A.B'])
        tm.assert_frame_equal(result.reindex_like(expected), expected)
        result = json_normalize({'A': {'A': 1, 'B': 2}}, sep='_')
        expected = DataFrame([[1, 2]], columns=['A_A', 'A_B'])
        tm.assert_frame_equal(result.reindex_like(expected), expected)
        result = json_normalize({'A': {'A': 1, 'B': 2}}, sep='σ')
        expected = DataFrame([[1, 2]], columns=['AσA', 'AσB'])
        tm.assert_frame_equal(result.reindex_like(expected), expected)
        result = json_normalize(deep_nested, ['states', 'cities'], meta=['country', ['states', 'name']], sep='_')
        expected = Index(['name', 'pop', 'country', 'states_name']).sort_values()
        assert result.columns.sort_values().equals(expected)

    def test_normalize_with_multichar_separator(self):
        data = {'a': [1, 2], 'b': {'b_1': 2, 'b_2': (3, 4)}}
        result = json_normalize(data, sep='__')
        expected = DataFrame([[[1, 2], 2, (3, 4)]], columns=['a', 'b__b_1', 'b__b_2'])
        tm.assert_frame_equal(result, expected)

    def test_value_array_record_prefix(self):
        result = json_normalize({'A': [1, 2]}, 'A', record_prefix='Prefix.')
        expected = DataFrame([[1], [2]], columns=['Prefix.0'])
        tm.assert_frame_equal(result, expected)

    def test_nested_object_record_path(self):
        data = {'state': 'Florida', 'info': {'governor': 'Rick Scott', 'counties': [{'name': 'Dade', 'population': 12345}, {'name': 'Broward', 'population': 40000}, {'name': 'Palm Beach', 'population': 60000}]}}
        result = json_normalize(data, record_path=['info', 'counties'])
        expected = DataFrame([['Dade', 12345], ['Broward', 40000], ['Palm Beach', 60000]], columns=['name', 'population'])
        tm.assert_frame_equal(result, expected)

    def test_more_deeply_nested(self, deep_nested):
        result = json_normalize(deep_nested, ['states', 'cities'], meta=['country', ['states', 'name']])
        ex_data = {'country': ['USA'] * 4 + ['Germany'] * 3, 'states.name': ['California', 'California', 'Ohio', 'Ohio', 'Bayern', 'Nordrhein-Westfalen', 'Nordrhein-Westfalen'], 'name': ['San Francisco', 'Los Angeles', 'Columbus', 'Cleveland', 'Munich', 'Duesseldorf', 'Koeln'], 'pop': [12345, 12346, 1234, 1236, 12347, 1238, 1239]}
        expected = DataFrame(ex_data, columns=result.columns)
        tm.assert_frame_equal(result, expected)

    def test_shallow_nested(self):
        data = [{'state': 'Florida', 'shortname': 'FL', 'info': {'governor': 'Rick Scott'}, 'counties': [{'name': 'Dade', 'population': 12345}, {'name': 'Broward', 'population': 40000}, {'name': 'Palm Beach', 'population': 60000}]}, {'state': 'Ohio', 'shortname': 'OH', 'info': {'governor': 'John Kasich'}, 'counties': [{'name': 'Summit', 'population': 1234}, {'name': 'Cuyahoga', 'population': 1337}]}]
        result = json_normalize(data, 'counties', ['state', 'shortname', ['info', 'governor']])
        ex_data = {'name': ['Dade', 'Broward', 'Palm Beach', 'Summit', 'Cuyahoga'], 'state': ['Florida'] * 3 + ['Ohio'] * 2, 'shortname': ['FL', 'FL', 'FL', 'OH', 'OH'], 'info.governor': ['Rick Scott'] * 3 + ['John Kasich'] * 2, 'population': [12345, 40000, 60000, 1234, 1337]}
        expected = DataFrame(ex_data, columns=result.columns)
        tm.assert_frame_equal(result, expected)

    def test_nested_meta_path_with_nested_record_path(self, state_data):
        result = json_normalize(data=state_data, record_path=['counties'], meta=['state', 'shortname', ['info', 'governor']], errors='ignore')
        ex_data = {'name': ['Dade', 'Broward', 'Palm Beach', 'Summit', 'Cuyahoga'], 'population': [12345, 40000, 60000, 1234, 1337], 'state': ['Florida'] * 3 + ['Ohio'] * 2, 'shortname': ['FL'] * 3 + ['OH'] * 2, 'info.governor': ['Rick Scott'] * 3 + ['John Kasich'] * 2}
        expected = DataFrame(ex_data)
        tm.assert_frame_equal(result, expected)

    def test_meta_name_conflict(self):
        data = [{'foo': 'hello', 'bar': 'there', 'data': [{'foo': 'something', 'bar': 'else'}, {'foo': 'something2', 'bar': 'else2'}]}]
        msg = 'Conflicting metadata name (foo|bar), need distinguishing prefix'
        with pytest.raises(ValueError, match=msg):
            json_normalize(data, 'data', meta=['foo', 'bar'])
        result = json_normalize(data, 'data', meta=['foo', 'bar'], meta_prefix='meta')
        for val in ['metafoo', 'metabar', 'foo', 'bar']:
            assert val in result

    def test_meta_parameter_not_modified(self):
        data = [{'foo': 'hello', 'bar': 'there', 'data': [{'foo': 'something', 'bar': 'else'}, {'foo': 'something2', 'bar': 'else2'}]}]
        COLUMNS = ['foo', 'bar']
        result = json_normalize(data, 'data', meta=COLUMNS, meta_prefix='meta')
        assert COLUMNS == ['foo', 'bar']
        for val in ['metafoo', 'metabar', 'foo', 'bar']:
            assert val in result

    def test_record_prefix(self, state_data):
        result = json_normalize(state_data[0], 'counties')
        expected = DataFrame(state_data[0]['counties'])
        tm.assert_frame_equal(result, expected)
        result = json_normalize(state_data, 'counties', meta='state', record_prefix='county_')
        expected = []
        for rec in state_data:
            expected.extend(rec['counties'])
        expected = DataFrame(expected)
        expected = expected.rename(columns=lambda x: 'county_' + x)
        expected['state'] = np.array(['Florida', 'Ohio']).repeat([3, 2])
        tm.assert_frame_equal(result, expected)

    def test_non_ascii_key(self):
        testjson = b'[{"\xc3\x9cnic\xc3\xb8de":0,"sub":{"A":1, "B":2}},{"\xc3\x9cnic\xc3\xb8de":1,"sub":{"A":3, "B":4}}]'.decode('utf8')
        testdata = {b'\xc3\x9cnic\xc3\xb8de'.decode('utf8'): [0, 1], 'sub.A': [1, 3], 'sub.B': [2, 4]}
        expected = DataFrame(testdata)
        result = json_normalize(json.loads(testjson))
        tm.assert_frame_equal(result, expected)

    def test_missing_field(self, author_missing_data):
        result = json_normalize(author_missing_data)
        ex_data = [{'info': np.nan, 'info.created_at': np.nan, 'info.last_updated': np.nan, 'author_name.first': np.nan, 'author_name.last_name': np.nan}, {'info': None, 'info.created_at': '11/08/1993', 'info.last_updated': '26/05/2012', 'author_name.first': 'Jane', 'author_name.last_name': 'Doe'}]
        expected = DataFrame(ex_data)
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize('max_level,expected', [(0, [{'TextField': 'Some text', 'UserField': {'Id': 'ID001', 'Name': 'Name001'}, 'CreatedBy': {'Name': 'User001'}, 'Image': {'a': 'b'}}, {'TextField': 'Some text', 'UserField': {'Id': 'ID001', 'Name': 'Name001'}, 'CreatedBy': {'Name': 'User001'}, 'Image': {'a': 'b'}}]), (1, [{'TextField': 'Some text', 'UserField.Id': 'ID001', 'UserField.Name': 'Name001', 'CreatedBy': {'Name': 'User001'}, 'Image': {'a': 'b'}}, {'TextField': 'Some text', 'UserField.Id': 'ID001', 'UserField.Name': 'Name001', 'CreatedBy': {'Name': 'User001'}, 'Image': {'a': 'b'}}])])
    def test_max_level_with_records_path(self, max_level, expected):
        test_input = [{'CreatedBy': {'Name': 'User001'}, 'Lookup': [{'TextField': 'Some text', 'UserField': {'Id': 'ID001', 'Name': 'Name001'}}, {'TextField': 'Some text', 'UserField': {'Id': 'ID001', 'Name': 'Name001'}}], 'Image': {'a': 'b'}, 'tags': [{'foo': 'something', 'bar': 'else'}, {'foo': 'something2', 'bar': 'else2'}]}]
        result = json_normalize(test_input, record_path=['Lookup'], meta=[['CreatedBy'], ['Image']], max_level=max_level)
        expected_df = DataFrame(data=expected, columns=result.columns.values)
        tm.assert_equal(expected_df, result)

    def test_nested_flattening_consistent(self):
        df1 = json_normalize([{'A': {'B': 1}}])
        df2 = json_normalize({'dummy': [{'A': {'B': 1}}]}, 'dummy')
        tm.assert_frame_equal(df1, df2)

    def test_nonetype_record_path(self, nulls_fixture):
        result = json_normalize([{'state': 'Texas', 'info': nulls_fixture}, {'state': 'Florida', 'info': [{'i': 2}]}], record_path=['info'])
        expected = DataFrame({'i': 2}, index=[0])
        tm.assert_equal(result, expected)

    @pytest.mark.parametrize('value', ['false', 'true', '{}', '1', '"text"'])
    def test_non_list_record_path_errors(self, value):
        parsed_value = json.loads(value)
        test_input = {'state': 'Texas', 'info': parsed_value}
        test_path = 'info'
        msg = f'{test_input} has non list value {parsed_value} for path {test_path}. Must be list or null.'
        with pytest.raises(TypeError, match=msg):
            json_normalize([test_input], record_path=[test_path])

    def test_meta_non_iterable(self):
        data = '[{"id": 99, "data": [{"one": 1, "two": 2}]}]'
        result = json_normalize(json.loads(data), record_path=['data'], meta=['id'])
        expected = DataFrame({'one': [1], 'two': [2], 'id': np.array([99], dtype=object)})
        tm.assert_frame_equal(result, expected)

    def test_generator(self, state_data):

        def generator_data():
            yield from state_data[0]['counties']
        result = json_normalize(generator_data())
        expected = DataFrame(state_data[0]['counties'])
        tm.assert_frame_equal(result, expected)

    def test_top_column_with_leading_underscore(self):
        data = {'_id': {'a1': 10, 'l2': {'l3': 0}}, 'gg': 4}
        result = json_normalize(data, sep='_')
        expected = DataFrame([[4, 10, 0]], columns=['gg', '_id_a1', '_id_l2_l3'])
        tm.assert_frame_equal(result, expected)