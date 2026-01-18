from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import io
import sys
from boto import config
from gslib.cloud_api import EncryptionException
from gslib.exception import CommandException
from gslib.exception import NO_URLS_MATCHED_TARGET
from gslib.storage_url import StorageUrlFromString
from gslib.utils.encryption_helper import CryptoKeyWrapperFromKey
from gslib.utils.encryption_helper import FindMatchingCSEKInBotoConfig
from gslib.utils.metadata_util import ObjectIsGzipEncoded
from gslib.utils import text_util
def CatUrlStrings(self, url_strings, show_header=False, start_byte=0, end_byte=None, cat_out_fd=None):
    """Prints each of the url strings to stdout.

    Args:
      url_strings: String iterable.
      show_header: If true, print a header per file.
      start_byte: Starting byte of the file to print, used for constructing
                  range requests.
      end_byte: Ending byte of the file to print; used for constructing range
                requests. If this is negative, the start_byte is ignored and
                and end range is sent over HTTP (such as range: bytes -9)
      cat_out_fd: File descriptor to which output should be written. Defaults to
                 stdout if no file descriptor is supplied.
    Returns:
      0 on success.

    Raises:
      CommandException if no URLs can be found.
    """
    printed_one = False
    if cat_out_fd is None:
        cat_out_fd = sys.stdout
    old_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        if url_strings and url_strings[0] in ('-', 'file://-'):
            self._WriteBytesBufferedFileToFile(sys.stdin, cat_out_fd)
        else:
            for url_str in url_strings:
                did_some_work = False
                for blr in self.command_obj.WildcardIterator(url_str).IterObjects(bucket_listing_fields=_CAT_BUCKET_LISTING_FIELDS):
                    decryption_keywrapper = None
                    if blr.root_object and blr.root_object.customerEncryption and blr.root_object.customerEncryption.keySha256:
                        decryption_key = FindMatchingCSEKInBotoConfig(blr.root_object.customerEncryption.keySha256, config)
                        if not decryption_key:
                            raise EncryptionException('Missing decryption key with SHA256 hash %s. No decryption key matches object %s' % (blr.root_object.customerEncryption.keySha256, blr.url_string))
                        decryption_keywrapper = CryptoKeyWrapperFromKey(decryption_key)
                    did_some_work = True
                    if show_header:
                        if printed_one:
                            print()
                        print('==> %s <==' % blr)
                        printed_one = True
                    cat_object = blr.root_object
                    if 0 < getattr(cat_object, 'size', -1) <= start_byte:
                        return 0
                    storage_url = StorageUrlFromString(blr.url_string)
                    if storage_url.IsCloudUrl():
                        compressed_encoding = ObjectIsGzipEncoded(cat_object)
                        self.command_obj.gsutil_api.GetObjectMedia(cat_object.bucket, cat_object.name, cat_out_fd, compressed_encoding=compressed_encoding, start_byte=start_byte, end_byte=end_byte, object_size=cat_object.size, generation=storage_url.generation, decryption_tuple=decryption_keywrapper, provider=storage_url.scheme)
                        cat_out_fd.flush()
                    else:
                        with open(storage_url.object_name, 'rb') as f:
                            self._WriteBytesBufferedFileToFile(f, cat_out_fd)
                if not did_some_work:
                    raise CommandException(NO_URLS_MATCHED_TARGET % url_str)
    finally:
        sys.stdout = old_stdout
    return 0