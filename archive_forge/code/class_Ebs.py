from boto.ec2.elb.listelement import ListElement
from boto.ec2.blockdevicemapping import BlockDeviceMapping as BDM
from boto.resultset import ResultSet
import boto.utils
import base64
class Ebs(object):

    def __init__(self, connection=None, snapshot_id=None, volume_size=None):
        self.connection = connection
        self.snapshot_id = snapshot_id
        self.volume_size = volume_size

    def __repr__(self):
        return 'Ebs(%s, %s)' % (self.snapshot_id, self.volume_size)

    def startElement(self, name, attrs, connection):
        pass

    def endElement(self, name, value, connection):
        if name == 'SnapshotId':
            self.snapshot_id = value
        elif name == 'VolumeSize':
            self.volume_size = value