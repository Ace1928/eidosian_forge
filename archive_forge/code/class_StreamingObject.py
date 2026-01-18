from boto.s3.key import Key
class StreamingObject(Object):

    def url(self, scheme='rtmp'):
        return super(StreamingObject, self).url(scheme)