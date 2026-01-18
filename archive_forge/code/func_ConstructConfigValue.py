from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from . import services_util
from apitools.base.py import encoding
def ConstructConfigValue(self, value_type):
    """Make a value to insert into the GenerateConfigReport request.

    Args:
      value_type: The type to encode the message into. Generally, either
        OldConfigValue or NewConfigValue.

    Returns:
      The encoded config value object of type value_type.
    """
    result = {}
    if not self.IsReadyForReport():
        return None
    elif self.config:
        result.update(self.config)
    elif self.swagger_path:
        config_file = self.messages.ConfigFile(filePath=self.swagger_path, fileContents=self.swagger_contents, fileType=self.messages.ConfigFile.FileTypeValueValuesEnum.OPEN_API_YAML)
        config_source_message = self.messages.ConfigSource(files=[config_file])
        result.update(encoding.MessageToDict(config_source_message))
    else:
        if self.config_id:
            resource = 'services/{0}/configs/{1}'.format(self.service, self.config_id)
        else:
            active_config_ids = services_util.GetActiveServiceConfigIdsForService(self.service)
            if active_config_ids:
                resource = 'services/{0}/configs/{1}'.format(self.service, active_config_ids[0])
            else:
                resource = 'services/{0}'.format(self.service)
        result.update({'name': resource})
    result.update({'@type': self.GetTypeUrl()})
    return encoding.DictToMessage(result, value_type)