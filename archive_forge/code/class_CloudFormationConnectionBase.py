import unittest
from datetime import datetime
from mock import Mock
from tests.unit import AWSMockServiceTestCase
from boto.cloudformation.connection import CloudFormationConnection
from boto.exception import BotoServerError
from boto.compat import json
class CloudFormationConnectionBase(AWSMockServiceTestCase):
    connection_class = CloudFormationConnection

    def setUp(self):
        super(CloudFormationConnectionBase, self).setUp()
        self.stack_id = u'arn:aws:cloudformation:us-east-1:18:stack/Name/id'