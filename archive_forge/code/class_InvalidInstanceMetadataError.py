import base64
import xml.sax
import boto
from boto import handler
from boto.compat import json, StandardError
from boto.resultset import ResultSet
class InvalidInstanceMetadataError(Exception):
    MSG = "You can set the 'metadata_service_num_attempts' in your boto config file to increase the number of times boto will attempt to retrieve credentials from the instance metadata service."

    def __init__(self, msg):
        final_msg = msg + '\n' + self.MSG
        super(InvalidInstanceMetadataError, self).__init__(final_msg)