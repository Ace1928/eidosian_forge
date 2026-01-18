import itertools
import os
import sys
import tempfile
import unittest
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from ...testing import assert_arrays_equal
from ..array_sequence import ArraySequence, concatenate, is_array_sequence
class TestArraySequence(unittest.TestCase):

    def test_creating_empty_arraysequence(self):
        check_empty_arr_seq(ArraySequence())

    def test_creating_arraysequence_from_list(self):
        check_empty_arr_seq(ArraySequence([]))
        N = 5
        for ndim in range(1, N + 1):
            common_shape = tuple([SEQ_DATA['rng'].randint(1, 10) for _ in range(ndim - 1)])
            data = generate_data(nb_arrays=5, common_shape=common_shape, rng=SEQ_DATA['rng'])
            check_arr_seq(ArraySequence(data), data)
        buffer_size = 1.0 / 1024 ** 2
        check_arr_seq(ArraySequence(iter(SEQ_DATA['data']), buffer_size), SEQ_DATA['data'])

    def test_creating_arraysequence_from_generator(self):
        gen_1, gen_2 = itertools.tee((e for e in SEQ_DATA['data']))
        seq = ArraySequence(gen_1)
        seq_with_buffer = ArraySequence(gen_2, buffer_size=256)
        assert seq_with_buffer.get_data().shape == seq.get_data().shape
        assert seq_with_buffer._buffer_size > seq._buffer_size
        check_arr_seq(seq, SEQ_DATA['data'])
        check_arr_seq(seq_with_buffer, SEQ_DATA['data'])
        check_empty_arr_seq(ArraySequence(gen_1))

    def test_creating_arraysequence_from_arraysequence(self):
        seq = ArraySequence(SEQ_DATA['data'])
        check_arr_seq(ArraySequence(seq), SEQ_DATA['data'])
        seq = ArraySequence()
        check_empty_arr_seq(ArraySequence(seq))

    def test_arraysequence_iter(self):
        assert_arrays_equal(SEQ_DATA['seq'], SEQ_DATA['data'])
        seq = SEQ_DATA['seq'].copy()
        seq._lengths = seq._lengths[::2]
        with pytest.raises(ValueError):
            list(seq)

    def test_arraysequence_copy(self):
        orig = SEQ_DATA['seq']
        seq = orig.copy()
        n_rows = seq.total_nb_rows
        assert n_rows == orig.total_nb_rows
        assert_array_equal(seq._data, orig._data[:n_rows])
        assert seq._data is not orig._data
        assert_array_equal(seq._offsets, orig._offsets)
        assert seq._offsets is not orig._offsets
        assert_array_equal(seq._lengths, orig._lengths)
        assert seq._lengths is not orig._lengths
        assert seq.common_shape == orig.common_shape
        seq = orig[::2].copy()
        check_arr_seq(seq, SEQ_DATA['data'][::2])
        assert seq._data is not orig._data

    def test_arraysequence_append(self):
        element = generate_data(nb_arrays=1, common_shape=SEQ_DATA['seq'].common_shape, rng=SEQ_DATA['rng'])[0]
        seq = SEQ_DATA['seq'].copy()
        seq.append(element)
        check_arr_seq(seq, SEQ_DATA['data'] + [element])
        seq = SEQ_DATA['seq'].copy()
        seq.append(element.tolist())
        check_arr_seq(seq, SEQ_DATA['data'] + [element])
        seq = ArraySequence()
        seq.append(element)
        check_arr_seq(seq, [element])
        seq = SEQ_DATA['seq'].copy()
        seq.append([])
        check_arr_seq(seq, SEQ_DATA['seq'])
        element = generate_data(nb_arrays=1, common_shape=SEQ_DATA['seq'].common_shape * 2, rng=SEQ_DATA['rng'])[0]
        with pytest.raises(ValueError):
            seq.append(element)

    def test_arraysequence_extend(self):
        new_data = generate_data(nb_arrays=10, common_shape=SEQ_DATA['seq'].common_shape, rng=SEQ_DATA['rng'])
        seq = SEQ_DATA['seq'].copy()
        seq.extend([])
        check_arr_seq(seq, SEQ_DATA['data'])
        seq = SEQ_DATA['seq'].copy()
        seq.extend(new_data)
        check_arr_seq(seq, SEQ_DATA['data'] + new_data)
        seq = SEQ_DATA['seq'].copy()
        seq.extend((d for d in new_data))
        check_arr_seq(seq, SEQ_DATA['data'] + new_data)
        seq = SEQ_DATA['seq'].copy()
        seq.extend(ArraySequence(new_data))
        check_arr_seq(seq, SEQ_DATA['data'] + new_data)
        seq = SEQ_DATA['seq'].copy()
        seq.extend(ArraySequence(new_data)[::2])
        check_arr_seq(seq, SEQ_DATA['data'] + new_data[::2])
        seq = ArraySequence()
        seq.extend(ArraySequence())
        check_empty_arr_seq(seq)
        seq.extend(SEQ_DATA['seq'])
        check_arr_seq(seq, SEQ_DATA['data'])
        data = generate_data(nb_arrays=10, common_shape=SEQ_DATA['seq'].common_shape * 2, rng=SEQ_DATA['rng'])
        seq = SEQ_DATA['seq'].copy()
        with pytest.raises(ValueError):
            seq.extend(data)
        working_slice = seq[:2]
        seq.extend(ArraySequence(new_data))

    def test_arraysequence_getitem(self):
        for i, e in enumerate(SEQ_DATA['seq']):
            assert_array_equal(SEQ_DATA['seq'][i], e)
        indices = list(range(len(SEQ_DATA['seq'])))
        seq_view = SEQ_DATA['seq'][indices]
        check_arr_seq_view(seq_view, SEQ_DATA['seq'])
        check_arr_seq(seq_view, SEQ_DATA['seq'])
        for dtype in [np.int8, np.int16, np.int32, np.int64]:
            seq_view = SEQ_DATA['seq'][np.array(indices, dtype=dtype)]
            check_arr_seq_view(seq_view, SEQ_DATA['seq'])
            check_arr_seq(seq_view, SEQ_DATA['seq'])
        SEQ_DATA['rng'].shuffle(indices)
        seq_view = SEQ_DATA['seq'][indices]
        check_arr_seq_view(seq_view, SEQ_DATA['seq'])
        check_arr_seq(seq_view, [SEQ_DATA['data'][i] for i in indices])
        seq_view = SEQ_DATA['seq'][::2]
        check_arr_seq_view(seq_view, SEQ_DATA['seq'])
        check_arr_seq(seq_view, SEQ_DATA['data'][::2])
        selection = np.array([False, True, True, False, True])
        seq_view = SEQ_DATA['seq'][selection]
        check_arr_seq_view(seq_view, SEQ_DATA['seq'])
        check_arr_seq(seq_view, [SEQ_DATA['data'][i] for i, keep in enumerate(selection) if keep])
        with pytest.raises(TypeError):
            SEQ_DATA['seq']['abc']
        seq_view = SEQ_DATA['seq'][:, 2]
        check_arr_seq_view(seq_view, SEQ_DATA['seq'])
        check_arr_seq(seq_view, [d[:, 2] for d in SEQ_DATA['data']])
        seq_view = SEQ_DATA['seq'][::-2][:, 2]
        check_arr_seq_view(seq_view, SEQ_DATA['seq'])
        check_arr_seq(seq_view, [d[:, 2] for d in SEQ_DATA['data'][::-2]])

    def test_arraysequence_setitem(self):
        seq = SEQ_DATA['seq'] * 0
        for i, e in enumerate(SEQ_DATA['seq']):
            seq[i] = e
        check_arr_seq(seq, SEQ_DATA['seq'])
        seq = SEQ_DATA['seq'].copy()
        seq[:] = 0
        assert seq._data.sum() == 0
        seq = SEQ_DATA['seq'] * 0
        seq[:] = SEQ_DATA['data']
        check_arr_seq(seq, SEQ_DATA['data'])
        seq = ArraySequence(np.arange(900).reshape((50, 6, 3)))
        seq[:, 0] = 0
        assert seq._data[:, 0].sum() == 0
        seq = ArraySequence(np.arange(900).reshape((50, 6, 3)))
        seq[range(len(seq))] = 0
        assert seq._data.sum() == 0
        seq = ArraySequence(np.arange(900).reshape((50, 6, 3)))
        seq[0:4] = seq[5:9]
        check_arr_seq(seq[0:4], seq[5:9])
        seq = ArraySequence(np.arange(900).reshape((50, 6, 3)))
        with pytest.raises(ValueError):
            seq[0:4] = seq[5:10]
        seq1 = ArraySequence(np.arange(10).reshape(5, 2))
        seq2 = ArraySequence(np.arange(15).reshape(5, 3))
        with pytest.raises(ValueError):
            seq1[0:5] = seq2
        seq1 = ArraySequence(np.arange(12).reshape(2, 2, 3))
        seq2 = ArraySequence(np.arange(8).reshape(2, 2, 2))
        with pytest.raises(ValueError):
            seq1[0:2] = seq2
        with pytest.raises(TypeError):
            seq[object()] = None

    def test_arraysequence_operators(self):
        flags = np.seterr(divide='ignore', invalid='ignore')
        SCALARS = [42, 0.5, True, -3, 0]
        CMP_OPS = ['__eq__', '__ne__', '__lt__', '__le__', '__gt__', '__ge__']
        seq = SEQ_DATA['seq'].copy()
        seq_int = SEQ_DATA['seq'].copy()
        seq_int._data = seq_int._data.astype(int)
        seq_bool = SEQ_DATA['seq'].copy() > 30
        ARRSEQS = [seq, seq_int, seq_bool]
        VIEWS = [seq[::2], seq_int[::2], seq_bool[::2]]

        def _test_unary(op, arrseq):
            orig = arrseq.copy()
            seq = getattr(orig, op)()
            assert seq is not orig
            check_arr_seq(seq, [getattr(d, op)() for d in orig])

        def _test_binary(op, arrseq, scalars, seqs, inplace=False):
            for scalar in scalars:
                orig = arrseq.copy()
                seq = getattr(orig, op)(scalar)
                assert (seq is orig) == inplace
                check_arr_seq(seq, [getattr(e, op)(scalar) for e in arrseq])
            for other in seqs:
                orig = arrseq.copy()
                seq = getattr(orig, op)(other)
                assert seq is not SEQ_DATA['seq']
                check_arr_seq(seq, [getattr(e1, op)(e2) for e1, e2 in zip(arrseq, other)])
            orig = arrseq.copy()
            with pytest.raises(ValueError):
                getattr(orig, op)(orig[::2])
            seq1 = ArraySequence(np.arange(10).reshape(5, 2))
            seq2 = ArraySequence(np.arange(15).reshape(5, 3))
            with pytest.raises(ValueError):
                getattr(seq1, op)(seq2)
            seq1 = ArraySequence(np.arange(12).reshape(2, 2, 3))
            seq2 = ArraySequence(np.arange(8).reshape(2, 2, 2))
            with pytest.raises(ValueError):
                getattr(seq1, op)(seq2)
        for op in ['__add__', '__sub__', '__mul__', '__mod__', '__floordiv__', '__truediv__'] + CMP_OPS:
            _test_binary(op, seq, SCALARS, ARRSEQS)
            _test_binary(op, seq_int, SCALARS, ARRSEQS)
            _test_binary(op, seq[::2], SCALARS, VIEWS)
            _test_binary(op, seq_int[::2], SCALARS, VIEWS)
            if op in CMP_OPS:
                continue
            op = f'__i{op.strip('_')}__'
            _test_binary(op, seq, SCALARS, ARRSEQS, inplace=True)
            if op == '__itruediv__':
                continue
            _test_binary(op, seq_int, [42, -3, True, 0], [seq_int, seq_bool, -seq_int], inplace=True)
            with pytest.raises(TypeError):
                _test_binary(op, seq_int, [0.5], [], inplace=True)
            with pytest.raises(TypeError):
                _test_binary(op, seq_int, [], [seq], inplace=True)
        _test_binary('__pow__', seq, [42, -3, True, 0], [seq_int, seq_bool, -seq_int])
        _test_binary('__ipow__', seq, [42, -3, True, 0], [seq_int, seq_bool, -seq_int], inplace=True)
        with pytest.raises(ValueError):
            _test_binary('__pow__', seq_int, [-3], [])
        with pytest.raises(ValueError):
            _test_binary('__ipow__', seq_int, [-3], [], inplace=True)
        for scalar in SCALARS + ARRSEQS:
            seq_int_cp = seq_int.copy()
            with pytest.raises(TypeError):
                seq_int_cp /= scalar
        for op in ('__lshift__', '__rshift__', '__or__', '__and__', '__xor__'):
            _test_binary(op, seq_bool, [42, -3, True, 0], [seq_int, seq_bool, -seq_int])
            with pytest.raises(TypeError):
                _test_binary(op, seq_bool, [0.5], [])
            with pytest.raises(TypeError):
                _test_binary(op, seq, [], [seq])
        for op in ['__neg__', '__abs__']:
            _test_unary(op, seq)
            _test_unary(op, -seq)
            _test_unary(op, seq_int)
            _test_unary(op, -seq_int)
        _test_unary('__abs__', seq_bool)
        _test_unary('__invert__', seq_bool)
        with pytest.raises(TypeError):
            _test_unary('__invert__', seq)
        np.seterr(**flags)

    def test_arraysequence_repr(self):
        repr(SEQ_DATA['seq'])
        nb_arrays = 50
        seq = ArraySequence(generate_data(nb_arrays, common_shape=(1,), rng=SEQ_DATA['rng']))
        bkp_threshold = np.get_printoptions()['threshold']
        np.set_printoptions(threshold=nb_arrays * 2)
        txt1 = repr(seq)
        np.set_printoptions(threshold=nb_arrays // 2)
        txt2 = repr(seq)
        assert len(txt2) < len(txt1)
        np.set_printoptions(threshold=bkp_threshold)

    def test_save_and_load_arraysequence(self):
        with tempfile.TemporaryFile(mode='w+b', suffix='.npz') as f:
            seq = ArraySequence()
            seq.save(f)
            f.seek(0, os.SEEK_SET)
            loaded_seq = ArraySequence.load(f)
            assert_array_equal(loaded_seq._data, seq._data)
            assert_array_equal(loaded_seq._offsets, seq._offsets)
            assert_array_equal(loaded_seq._lengths, seq._lengths)
        with tempfile.TemporaryFile(mode='w+b', suffix='.npz') as f:
            seq = SEQ_DATA['seq']
            seq.save(f)
            f.seek(0, os.SEEK_SET)
            loaded_seq = ArraySequence.load(f)
            assert_array_equal(loaded_seq._data, seq._data)
            assert_array_equal(loaded_seq._offsets, seq._offsets)
            assert_array_equal(loaded_seq._lengths, seq._lengths)
            loaded_seq.append(SEQ_DATA['data'][0])

    def test_get_data(self):
        seq_view = SEQ_DATA['seq'][::2]
        check_arr_seq_view(seq_view, SEQ_DATA['seq'])
        data = seq_view.get_data()
        assert len(data) < len(seq_view._data)