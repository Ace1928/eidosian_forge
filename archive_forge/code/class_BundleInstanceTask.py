from boto.ec2.ec2object import EC2Object
class BundleInstanceTask(EC2Object):

    def __init__(self, connection=None):
        super(BundleInstanceTask, self).__init__(connection)
        self.id = None
        self.instance_id = None
        self.progress = None
        self.start_time = None
        self.state = None
        self.bucket = None
        self.prefix = None
        self.upload_policy = None
        self.upload_policy_signature = None
        self.update_time = None
        self.code = None
        self.message = None

    def __repr__(self):
        return 'BundleInstanceTask:%s' % self.id

    def startElement(self, name, attrs, connection):
        return None

    def endElement(self, name, value, connection):
        if name == 'bundleId':
            self.id = value
        elif name == 'instanceId':
            self.instance_id = value
        elif name == 'progress':
            self.progress = value
        elif name == 'startTime':
            self.start_time = value
        elif name == 'state':
            self.state = value
        elif name == 'bucket':
            self.bucket = value
        elif name == 'prefix':
            self.prefix = value
        elif name == 'uploadPolicy':
            self.upload_policy = value
        elif name == 'uploadPolicySignature':
            self.upload_policy_signature = value
        elif name == 'updateTime':
            self.update_time = value
        elif name == 'code':
            self.code = value
        elif name == 'message':
            self.message = value
        else:
            setattr(self, name, value)