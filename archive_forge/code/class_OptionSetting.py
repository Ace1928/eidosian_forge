from boto.rds.dbsecuritygroup import DBSecurityGroup
from boto.resultset import ResultSet
class OptionSetting(object):
    """
    Describes a OptionSetting for use in an Option

    :ivar name: The name of the option that has settings that you can set.
    :ivar description: The description of the option setting.
    :ivar value: The current value of the option setting.
    :ivar default_value: The default value of the option setting.
    :ivar allowed_values: The allowed values of the option setting.
    :ivar data_type: The data type of the option setting.
    :ivar apply_type: The DB engine specific parameter type.
    :ivar is_modifiable: A Boolean value that, when true, indicates the option
                         setting can be modified from the default.
    :ivar is_collection: Indicates if the option setting is part of a
                         collection.
    """

    def __init__(self, name=None, description=None, value=None, default_value=False, allowed_values=None, data_type=None, apply_type=None, is_modifiable=False, is_collection=False):
        self.name = name
        self.description = description
        self.value = value
        self.default_value = default_value
        self.allowed_values = allowed_values
        self.data_type = data_type
        self.apply_type = apply_type
        self.is_modifiable = is_modifiable
        self.is_collection = is_collection

    def __repr__(self):
        return 'OptionSetting:%s' % self.name

    def startElement(self, name, attrs, connection):
        return None

    def endElement(self, name, value, connection):
        if name == 'Name':
            self.name = value
        elif name == 'Description':
            self.description = value
        elif name == 'Value':
            self.value = value
        elif name == 'DefaultValue':
            self.default_value = value
        elif name == 'AllowedValues':
            self.allowed_values = value
        elif name == 'DataType':
            self.data_type = value
        elif name == 'ApplyType':
            self.apply_type = value
        elif name == 'IsModifiable':
            if value.lower() == 'true':
                self.is_modifiable = True
            else:
                self.is_modifiable = False
        elif name == 'IsCollection':
            if value.lower() == 'true':
                self.is_collection = True
            else:
                self.is_collection = False
        else:
            setattr(self, name, value)