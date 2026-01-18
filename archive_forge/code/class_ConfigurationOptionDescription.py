from datetime import datetime
from boto.compat import six
class ConfigurationOptionDescription(BaseObject):

    def __init__(self, response):
        super(ConfigurationOptionDescription, self).__init__()
        self.change_severity = str(response['ChangeSeverity'])
        self.default_value = str(response['DefaultValue'])
        self.max_length = int(response['MaxLength']) if response['MaxLength'] else None
        self.max_value = int(response['MaxValue']) if response['MaxValue'] else None
        self.min_value = int(response['MinValue']) if response['MinValue'] else None
        self.name = str(response['Name'])
        self.namespace = str(response['Namespace'])
        if response['Regex']:
            self.regex = OptionRestrictionRegex(response['Regex'])
        else:
            self.regex = None
        self.user_defined = str(response['UserDefined'])
        self.value_options = []
        if response['ValueOptions']:
            for member in response['ValueOptions']:
                value_option = str(member)
                self.value_options.append(value_option)
        self.value_type = str(response['ValueType'])