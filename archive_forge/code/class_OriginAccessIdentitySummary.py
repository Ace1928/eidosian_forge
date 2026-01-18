import uuid
class OriginAccessIdentitySummary(object):

    def __init__(self, connection=None, id='', s3_user_id='', comment=''):
        self.connection = connection
        self.id = id
        self.s3_user_id = s3_user_id
        self.comment = comment
        self.etag = None

    def startElement(self, name, attrs, connection):
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

    def get_origin_access_identity(self):
        return self.connection.get_origin_access_identity_info(self.id)