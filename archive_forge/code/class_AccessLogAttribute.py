class AccessLogAttribute(object):
    """
    Represents the AccessLog segment of ELB attributes.
    """

    def __init__(self, connection=None):
        self.enabled = None
        self.s3_bucket_name = None
        self.s3_bucket_prefix = None
        self.emit_interval = None

    def __repr__(self):
        return 'AccessLog(%s, %s, %s, %s)' % (self.enabled, self.s3_bucket_name, self.s3_bucket_prefix, self.emit_interval)

    def startElement(self, name, attrs, connection):
        pass

    def endElement(self, name, value, connection):
        if name == 'Enabled':
            if value.lower() == 'true':
                self.enabled = True
            else:
                self.enabled = False
        elif name == 'S3BucketName':
            self.s3_bucket_name = value
        elif name == 'S3BucketPrefix':
            self.s3_bucket_prefix = value
        elif name == 'EmitInterval':
            self.emit_interval = int(value)