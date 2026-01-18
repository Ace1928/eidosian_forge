import hashlib
from boto.glacier.utils import chunk_hashes, tree_hash, bytes_to_hex
from boto.glacier.utils import compute_hashes_from_fileobj
def _upload_part(self, part_data):
    self.uploader.upload_part(self.next_part_index, part_data)
    self.next_part_index += 1