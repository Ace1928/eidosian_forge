import codecs
from boto.glacier.exceptions import UploadArchiveError
from boto.glacier.job import Job
from boto.glacier.writer import compute_hashes_from_fileobj, \
from boto.glacier.concurrent import ConcurrentUploader
from boto.glacier.utils import minimum_part_size, DEFAULT_PART_SIZE
import os.path
@staticmethod
def _range_string_to_part_index(range_string, part_size):
    start, inside_end = [int(value) for value in range_string.split('-')]
    end = inside_end + 1
    length = end - start
    if length == part_size + 1:
        end -= 1
        inside_end -= 1
        length -= 1
    assert not start % part_size, 'upload part start byte is not on a part boundary'
    assert length <= part_size, 'upload part is bigger than part size'
    return start // part_size