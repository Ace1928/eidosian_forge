from array import array
from srsly.msgpack import packb, unpackb
def _runtest(format, nbytes, expected_header, expected_prefix, use_bin_type):
    original_array = array(format)
    original_array.fromlist([255] * (nbytes // original_array.itemsize))
    original_data = get_data(original_array)
    view = make_memoryview(original_array)
    packed = packb(view, use_bin_type=use_bin_type)
    unpacked = unpackb(packed)
    reconstructed_array = make_array(format, unpacked)
    assert len(original_data) == nbytes
    assert packed[:1] == expected_header
    assert packed[1:1 + len(expected_prefix)] == expected_prefix
    assert packed[1 + len(expected_prefix):] == original_data
    assert original_array == reconstructed_array