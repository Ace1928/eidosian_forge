from datetime import datetime
from boto.compat import six
class ValidateConfigurationSettingsResponse(Response):

    def __init__(self, response):
        response = response['ValidateConfigurationSettingsResponse']
        super(ValidateConfigurationSettingsResponse, self).__init__(response)
        response = response['ValidateConfigurationSettingsResult']
        self.messages = []
        if response['Messages']:
            for member in response['Messages']:
                message = ValidationMessage(member)
                self.messages.append(message)