class AccountLimits(object):

    def __init__(self, connection=None):
        self.connection = connection
        self.max_autoscaling_groups = None
        self.max_launch_configurations = None

    def __repr__(self):
        return 'AccountLimits: [%s, %s]' % (self.max_autoscaling_groups, self.max_launch_configurations)

    def startElement(self, name, attrs, connection):
        return None

    def endElement(self, name, value, connection):
        if name == 'RequestId':
            self.request_id = value
        elif name == 'MaxNumberOfAutoScalingGroups':
            self.max_autoscaling_groups = int(value)
        elif name == 'MaxNumberOfLaunchConfigurations':
            self.max_launch_configurations = int(value)
        else:
            setattr(self, name, value)