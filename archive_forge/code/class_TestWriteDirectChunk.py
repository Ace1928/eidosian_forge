import h5py
import numpy
import numpy.testing
import pytest
from .common import ut, TestCase
class TestWriteDirectChunk(TestCase):

    def test_write_direct_chunk(self):
        filename = self.mktemp().encode()
        with h5py.File(filename, 'w') as filehandle:
            dataset = filehandle.create_dataset('data', (100, 100, 100), maxshape=(None, 100, 100), chunks=(1, 100, 100), dtype='float32')
            array = numpy.zeros((10, 100, 100))
            for index in range(10):
                a = numpy.random.rand(100, 100).astype('float32')
                dataset.id.write_direct_chunk((index, 0, 0), a.tobytes(), filter_mask=1)
                array[index] = a
        with h5py.File(filename, 'r') as filehandle:
            for i in range(10):
                read_data = filehandle['data'][i]
                numpy.testing.assert_array_equal(array[i], read_data)