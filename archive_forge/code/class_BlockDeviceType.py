class BlockDeviceType(object):
    """
    Represents parameters for a block device.
    """

    def __init__(self, connection=None, ephemeral_name=None, no_device=False, volume_id=None, snapshot_id=None, status=None, attach_time=None, delete_on_termination=False, size=None, volume_type=None, iops=None, encrypted=None):
        self.connection = connection
        self.ephemeral_name = ephemeral_name
        self.no_device = no_device
        self.volume_id = volume_id
        self.snapshot_id = snapshot_id
        self.status = status
        self.attach_time = attach_time
        self.delete_on_termination = delete_on_termination
        self.size = size
        self.volume_type = volume_type
        self.iops = iops
        self.encrypted = encrypted

    def startElement(self, name, attrs, connection):
        pass

    def endElement(self, name, value, connection):
        lname = name.lower()
        if name == 'volumeId':
            self.volume_id = value
        elif lname == 'virtualname':
            self.ephemeral_name = value
        elif lname == 'nodevice':
            self.no_device = value == 'true'
        elif lname == 'snapshotid':
            self.snapshot_id = value
        elif lname == 'volumesize':
            self.size = int(value)
        elif lname == 'status':
            self.status = value
        elif lname == 'attachtime':
            self.attach_time = value
        elif lname == 'deleteontermination':
            self.delete_on_termination = value == 'true'
        elif lname == 'volumetype':
            self.volume_type = value
        elif lname == 'iops':
            self.iops = int(value)
        elif lname == 'encrypted':
            self.encrypted = value == 'true'
        else:
            setattr(self, name, value)