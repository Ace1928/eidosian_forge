import math
import socket
from boto.glacier.exceptions import TreeHashDoesNotMatchError, \
from boto.glacier.utils import tree_hash_from_str
def _calc_num_chunks(self, chunk_size):
    return int(math.ceil(self.archive_size / float(chunk_size)))