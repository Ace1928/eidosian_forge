import hashlib
from oslo_utils.secretutils import md5
from oslotest import base
import glance_store.driver as driver
class _FakeStore(object):

    @driver.back_compat_add
    def add(self, image_id, image_file, image_size, hashing_algo, context=None, verifier=None):
        """This is a 0.26.0+ add, returns a 5-tuple"""
        if hashing_algo == 'md5':
            hasher = md5(usedforsecurity=False)
        else:
            hasher = hashlib.new(str(hashing_algo))
        hasher.update(image_file)
        backend_url = 'backend://%s' % image_id
        bytes_written = len(image_file)
        checksum = md5(image_file, usedforsecurity=False).hexdigest()
        multihash = hasher.hexdigest()
        metadata_dict = {'verifier_obj': verifier.name if verifier else None, 'context_obj': context.name if context else None}
        return (backend_url, bytes_written, checksum, multihash, metadata_dict)