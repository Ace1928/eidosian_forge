from boto.s3 import user
from boto.s3 import key
from boto import handler
import xml.sax
class CompleteMultiPartUpload(object):
    """
    Represents a completed MultiPart Upload.  Contains the
    following useful attributes:

     * location - The URI of the completed upload
     * bucket_name - The name of the bucket in which the upload
                     is contained
     * key_name - The name of the new, completed key
     * etag - The MD5 hash of the completed, combined upload
     * version_id - The version_id of the completed upload
     * encrypted - The value of the encryption header
    """

    def __init__(self, bucket=None):
        self.bucket = bucket
        self.location = None
        self.bucket_name = None
        self.key_name = None
        self.etag = None
        self.version_id = None
        self.encrypted = None

    def __repr__(self):
        return '<CompleteMultiPartUpload: %s.%s>' % (self.bucket_name, self.key_name)

    def startElement(self, name, attrs, connection):
        return None

    def endElement(self, name, value, connection):
        if name == 'Location':
            self.location = value
        elif name == 'Bucket':
            self.bucket_name = value
        elif name == 'Key':
            self.key_name = value
        elif name == 'ETag':
            self.etag = value
        else:
            setattr(self, name, value)