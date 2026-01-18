import uuid
class OriginAccessIdentity(object):

    def __init__(self, connection=None, config=None, id='', s3_user_id='', comment=''):
        self.connection = connection
        self.config = config
        self.id = id
        self.s3_user_id = s3_user_id
        self.comment = comment
        self.etag = None

    def startElement(self, name, attrs, connection):
        if name == 'CloudFrontOriginAccessIdentityConfig':
            self.config = OriginAccessIdentityConfig()
            return self.config
        else:
            return None

    def endElement(self, name, value, connection):
        if name == 'Id':
            self.id = value
        elif name == 'S3CanonicalUserId':
            self.s3_user_id = value
        elif name == 'Comment':
            self.comment = value
        else:
            setattr(self, name, value)

    def update(self, comment=None):
        new_config = OriginAccessIdentityConfig(self.connection, self.config.caller_reference, self.config.comment)
        if comment is not None:
            new_config.comment = comment
        self.etag = self.connection.set_origin_identity_config(self.id, self.etag, new_config)
        self.config = new_config

    def delete(self):
        return self.connection.delete_origin_access_identity(self.id, self.etag)

    def uri(self):
        return 'origin-access-identity/cloudfront/%s' % self.id