from datetime import datetime
class Activity(object):

    def __init__(self, connection=None):
        self.connection = connection
        self.start_time = None
        self.end_time = None
        self.activity_id = None
        self.progress = None
        self.status_code = None
        self.cause = None
        self.description = None
        self.status_message = None
        self.group_name = None

    def __repr__(self):
        return 'Activity<%s>: For group:%s, progress:%s, cause:%s' % (self.activity_id, self.group_name, self.status_message, self.cause)

    def startElement(self, name, attrs, connection):
        return None

    def endElement(self, name, value, connection):
        if name == 'ActivityId':
            self.activity_id = value
        elif name == 'AutoScalingGroupName':
            self.group_name = value
        elif name == 'StartTime':
            try:
                self.start_time = datetime.strptime(value, '%Y-%m-%dT%H:%M:%S.%fZ')
            except ValueError:
                self.start_time = datetime.strptime(value, '%Y-%m-%dT%H:%M:%SZ')
        elif name == 'EndTime':
            try:
                self.end_time = datetime.strptime(value, '%Y-%m-%dT%H:%M:%S.%fZ')
            except ValueError:
                self.end_time = datetime.strptime(value, '%Y-%m-%dT%H:%M:%SZ')
        elif name == 'Progress':
            self.progress = value
        elif name == 'Cause':
            self.cause = value
        elif name == 'Description':
            self.description = value
        elif name == 'StatusMessage':
            self.status_message = value
        elif name == 'StatusCode':
            self.status_code = value
        else:
            setattr(self, name, value)