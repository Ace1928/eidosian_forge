from datetime import datetime
from boto.compat import six
class OptionRestrictionRegex(BaseObject):

    def __init__(self, response):
        super(OptionRestrictionRegex, self).__init__()
        self.label = response['Label']
        self.pattern = response['Pattern']