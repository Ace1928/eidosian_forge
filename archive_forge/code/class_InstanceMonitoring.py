from boto.ec2.elb.listelement import ListElement
from boto.ec2.blockdevicemapping import BlockDeviceMapping as BDM
from boto.resultset import ResultSet
import boto.utils
import base64
class InstanceMonitoring(object):

    def __init__(self, connection=None, enabled='false'):
        self.connection = connection
        self.enabled = enabled

    def __repr__(self):
        return 'InstanceMonitoring(%s)' % self.enabled

    def startElement(self, name, attrs, connection):
        pass

    def endElement(self, name, value, connection):
        if name == 'Enabled':
            self.enabled = value