import codecs
from boto.glacier.exceptions import UploadArchiveError
from boto.glacier.job import Job
from boto.glacier.writer import compute_hashes_from_fileobj, \
from boto.glacier.concurrent import ConcurrentUploader
from boto.glacier.utils import minimum_part_size, DEFAULT_PART_SIZE
import os.path
def _upload_archive_single_operation(self, filename, description):
    """
        Adds an archive to a vault in a single operation. It's recommended for
        archives less than 100MB

        :type file: str
        :param file: A filename to upload

        :type description: str
        :param description: A description for the archive.

        :rtype: str
        :return: The archive id of the newly created archive
        """
    with open(filename, 'rb') as fileobj:
        linear_hash, tree_hash = compute_hashes_from_fileobj(fileobj)
        fileobj.seek(0)
        response = self.layer1.upload_archive(self.name, fileobj, linear_hash, tree_hash, description)
    return response['ArchiveId']