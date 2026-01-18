from boto.sqs.message import MHMessage
from boto.utils import get_ts
from socket import gethostname
import os, mimetypes, time
class ServiceMessage(MHMessage):

    def for_key(self, key, params=None, bucket_name=None):
        if params:
            self.update(params)
        if key.path:
            t = os.path.split(key.path)
            self['OriginalLocation'] = t[0]
            self['OriginalFileName'] = t[1]
            mime_type = mimetypes.guess_type(t[1])[0]
            if mime_type is None:
                mime_type = 'application/octet-stream'
            self['Content-Type'] = mime_type
            s = os.stat(key.path)
            t = time.gmtime(s[7])
            self['FileAccessedDate'] = get_ts(t)
            t = time.gmtime(s[8])
            self['FileModifiedDate'] = get_ts(t)
            t = time.gmtime(s[9])
            self['FileCreateDate'] = get_ts(t)
        else:
            self['OriginalFileName'] = key.name
            self['OriginalLocation'] = key.bucket.name
            self['ContentType'] = key.content_type
        self['Host'] = gethostname()
        if bucket_name:
            self['Bucket'] = bucket_name
        else:
            self['Bucket'] = key.bucket.name
        self['InputKey'] = key.name
        self['Size'] = key.size