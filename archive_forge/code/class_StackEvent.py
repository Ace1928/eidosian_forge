from datetime import datetime
from boto.resultset import ResultSet
class StackEvent(object):
    valid_states = ('CREATE_IN_PROGRESS', 'CREATE_FAILED', 'CREATE_COMPLETE', 'DELETE_IN_PROGRESS', 'DELETE_FAILED', 'DELETE_COMPLETE')

    def __init__(self, connection=None):
        self.connection = connection
        self.event_id = None
        self.logical_resource_id = None
        self.physical_resource_id = None
        self.resource_properties = None
        self.resource_status = None
        self.resource_status_reason = None
        self.resource_type = None
        self.stack_id = None
        self.stack_name = None
        self.timestamp = None

    def startElement(self, name, attrs, connection):
        return None

    def endElement(self, name, value, connection):
        if name == 'EventId':
            self.event_id = value
        elif name == 'LogicalResourceId':
            self.logical_resource_id = value
        elif name == 'PhysicalResourceId':
            self.physical_resource_id = value
        elif name == 'ResourceProperties':
            self.resource_properties = value
        elif name == 'ResourceStatus':
            self.resource_status = value
        elif name == 'ResourceStatusReason':
            self.resource_status_reason = value
        elif name == 'ResourceType':
            self.resource_type = value
        elif name == 'StackId':
            self.stack_id = value
        elif name == 'StackName':
            self.stack_name = value
        elif name == 'Timestamp':
            try:
                self.timestamp = datetime.strptime(value, '%Y-%m-%dT%H:%M:%SZ')
            except ValueError:
                self.timestamp = datetime.strptime(value, '%Y-%m-%dT%H:%M:%S.%fZ')
        else:
            setattr(self, name, value)

    def __repr__(self):
        return 'StackEvent %s %s %s' % (self.resource_type, self.logical_resource_id, self.resource_status)