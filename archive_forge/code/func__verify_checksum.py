import io
from openstack import exceptions
from openstack import resource
from openstack import utils
def _verify_checksum(md5, checksum):
    if checksum:
        digest = md5.hexdigest()
        if digest != checksum:
            raise exceptions.InvalidResponse('checksum mismatch: %s != %s' % (checksum, digest))