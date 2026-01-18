from datetime import datetime
from boto.compat import six
class ApplicationVersionDescription(BaseObject):

    def __init__(self, response):
        super(ApplicationVersionDescription, self).__init__()
        self.application_name = str(response['ApplicationName'])
        self.date_created = datetime.fromtimestamp(response['DateCreated'])
        self.date_updated = datetime.fromtimestamp(response['DateUpdated'])
        self.description = str(response['Description'])
        if response['SourceBundle']:
            self.source_bundle = S3Location(response['SourceBundle'])
        else:
            self.source_bundle = None
        self.version_label = str(response['VersionLabel'])