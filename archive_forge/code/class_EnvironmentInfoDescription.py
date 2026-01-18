from datetime import datetime
from boto.compat import six
class EnvironmentInfoDescription(BaseObject):

    def __init__(self, response):
        super(EnvironmentInfoDescription, self).__init__()
        self.ec2_instance_id = str(response['Ec2InstanceId'])
        self.info_type = str(response['InfoType'])
        self.message = str(response['Message'])
        self.sample_timestamp = datetime.fromtimestamp(response['SampleTimestamp'])