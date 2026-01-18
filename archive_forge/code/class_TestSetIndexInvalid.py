from datetime import (
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
class TestSetIndexInvalid:

    def test_set_index_verify_integrity(self, frame_of_index_cols):
        df = frame_of_index_cols
        with pytest.raises(ValueError, match='Index has duplicate keys'):
            df.set_index('A', verify_integrity=True)
        with pytest.raises(ValueError, match='Index has duplicate keys'):
            df.set_index([df['A'], df['A']], verify_integrity=True)

    @pytest.mark.parametrize('append', [True, False])
    @pytest.mark.parametrize('drop', [True, False])
    def test_set_index_raise_keys(self, frame_of_index_cols, drop, append):
        df = frame_of_index_cols
        with pytest.raises(KeyError, match="['foo', 'bar', 'baz']"):
            df.set_index(['foo', 'bar', 'baz'], drop=drop, append=append)
        with pytest.raises(KeyError, match='X'):
            df.set_index([df['A'], df['B'], 'X'], drop=drop, append=append)
        msg = "[('foo', 'foo', 'foo', 'bar', 'bar')]"
        with pytest.raises(KeyError, match=msg):
            df.set_index(tuple(df['A']), drop=drop, append=append)
        with pytest.raises(KeyError, match=msg):
            df.set_index(['A', df['A'], tuple(df['A'])], drop=drop, append=append)

    @pytest.mark.parametrize('append', [True, False])
    @pytest.mark.parametrize('drop', [True, False])
    @pytest.mark.parametrize('box', [set], ids=['set'])
    def test_set_index_raise_on_type(self, frame_of_index_cols, box, drop, append):
        df = frame_of_index_cols
        msg = 'The parameter "keys" may be a column key, .*'
        with pytest.raises(TypeError, match=msg):
            df.set_index(box(df['A']), drop=drop, append=append)
        with pytest.raises(TypeError, match=msg):
            df.set_index(['A', df['A'], box(df['A'])], drop=drop, append=append)

    @pytest.mark.parametrize('box', [Series, Index, np.array, iter, lambda x: MultiIndex.from_arrays([x])], ids=['Series', 'Index', 'np.array', 'iter', 'MultiIndex'])
    @pytest.mark.parametrize('length', [4, 6], ids=['too_short', 'too_long'])
    @pytest.mark.parametrize('append', [True, False])
    @pytest.mark.parametrize('drop', [True, False])
    def test_set_index_raise_on_len(self, frame_of_index_cols, box, length, drop, append):
        df = frame_of_index_cols
        values = np.random.default_rng(2).integers(0, 10, (length,))
        msg = 'Length mismatch: Expected 5 rows, received array of length.*'
        with pytest.raises(ValueError, match=msg):
            df.set_index(box(values), drop=drop, append=append)
        with pytest.raises(ValueError, match=msg):
            df.set_index(['A', df.A, box(values)], drop=drop, append=append)