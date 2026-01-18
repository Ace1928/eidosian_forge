import hashlib
from boto.glacier.utils import chunk_hashes, tree_hash, bytes_to_hex
from boto.glacier.utils import compute_hashes_from_fileobj
class _Uploader(object):
    """Upload to a Glacier upload_id.

    Call upload_part for each part (in any order) and then close to complete
    the upload.

    """

    def __init__(self, vault, upload_id, part_size, chunk_size=_ONE_MEGABYTE):
        self.vault = vault
        self.upload_id = upload_id
        self.part_size = part_size
        self.chunk_size = chunk_size
        self.archive_id = None
        self._uploaded_size = 0
        self._tree_hashes = []
        self.closed = False

    def _insert_tree_hash(self, index, raw_tree_hash):
        list_length = len(self._tree_hashes)
        if index >= list_length:
            self._tree_hashes.extend([None] * (list_length - index + 1))
        self._tree_hashes[index] = raw_tree_hash

    def upload_part(self, part_index, part_data):
        """Upload a part to Glacier.

        :param part_index: part number where 0 is the first part
        :param part_data: data to upload corresponding to this part

        """
        if self.closed:
            raise ValueError('I/O operation on closed file')
        part_tree_hash = tree_hash(chunk_hashes(part_data, self.chunk_size))
        self._insert_tree_hash(part_index, part_tree_hash)
        hex_tree_hash = bytes_to_hex(part_tree_hash)
        linear_hash = hashlib.sha256(part_data).hexdigest()
        start = self.part_size * part_index
        content_range = (start, start + len(part_data) - 1)
        response = self.vault.layer1.upload_part(self.vault.name, self.upload_id, linear_hash, hex_tree_hash, content_range, part_data)
        response.read()
        self._uploaded_size += len(part_data)

    def skip_part(self, part_index, part_tree_hash, part_length):
        """Skip uploading of a part.

        The final close call needs to calculate the tree hash and total size
        of all uploaded data, so this is the mechanism for resume
        functionality to provide it without actually uploading the data again.

        :param part_index: part number where 0 is the first part
        :param part_tree_hash: binary tree_hash of part being skipped
        :param part_length: length of part being skipped

        """
        if self.closed:
            raise ValueError('I/O operation on closed file')
        self._insert_tree_hash(part_index, part_tree_hash)
        self._uploaded_size += part_length

    def close(self):
        if self.closed:
            return
        if None in self._tree_hashes:
            raise RuntimeError('Some parts were not uploaded.')
        hex_tree_hash = bytes_to_hex(tree_hash(self._tree_hashes))
        response = self.vault.layer1.complete_multipart_upload(self.vault.name, self.upload_id, hex_tree_hash, self._uploaded_size)
        self.archive_id = response['ArchiveId']
        self.closed = True