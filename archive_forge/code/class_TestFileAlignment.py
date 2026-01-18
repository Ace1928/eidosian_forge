import h5py
from .common import TestCase
class TestFileAlignment(TestCase):
    """
        Ensure that setting the file alignment has the desired effect
        in the internal structure.
    """

    def test_no_alignment_set(self):
        fname = self.mktemp()
        shape = (881,)
        with h5py.File(fname, 'w') as h5file:
            for i in range(1000):
                dataset = h5file.create_dataset(dataset_name(i), shape, dtype='uint8')
                dataset[...] = i
                if not is_aligned(dataset):
                    break
            else:
                raise RuntimeError('Data was all found to be aligned to 4096')

    def test_alignment_set_above_threshold(self):
        alignment_threshold = 1000
        alignment_interval = 4096
        for shape in [(1033,), (1000,), (1001,)]:
            fname = self.mktemp()
            with h5py.File(fname, 'w', alignment_threshold=alignment_threshold, alignment_interval=alignment_interval) as h5file:
                for i in range(1000):
                    dataset = h5file.create_dataset(dataset_name(i), shape, dtype='uint8')
                    dataset[...] = i % 256
                    assert is_aligned(dataset, offset=alignment_interval)

    def test_alignment_set_below_threshold(self):
        alignment_threshold = 1000
        alignment_interval = 1024
        for shape in [(881,), (999,)]:
            fname = self.mktemp()
            with h5py.File(fname, 'w', alignment_threshold=alignment_threshold, alignment_interval=alignment_interval) as h5file:
                for i in range(1000):
                    dataset = h5file.create_dataset(dataset_name(i), shape, dtype='uint8')
                    dataset[...] = i
                    if not is_aligned(dataset, offset=alignment_interval):
                        break
                else:
                    raise RuntimeError(f'Data was all found to be aligned to {alignment_interval}. This is highly unlikely.')