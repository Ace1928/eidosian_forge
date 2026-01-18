import json
from tests.unit import AWSMockServiceTestCase
from boto.beanstalk.layer1 import Layer1
class TestCreateApplicationVersion(AWSMockServiceTestCase):
    connection_class = Layer1

    def default_body(self):
        return json.dumps({'CreateApplicationVersionResponse': {u'CreateApplicationVersionResult': {u'ApplicationVersion': {u'ApplicationName': u'application1', u'DateCreated': 1343067094.342, u'DateUpdated': 1343067094.342, u'Description': None, u'SourceBundle': {u'S3Bucket': u'elasticbeanstalk-us-east-1', u'S3Key': u'resources/elasticbeanstalk-sampleapp.war'}, u'VersionLabel': u'version1'}}}}).encode('utf-8')

    def test_create_application_version(self):
        self.set_http_response(status_code=200)
        api_response = self.service_connection.create_application_version('application1', 'version1', s3_bucket='mybucket', s3_key='mykey', auto_create_application=True)
        app_version = api_response['CreateApplicationVersionResponse']['CreateApplicationVersionResult']['ApplicationVersion']
        self.assert_request_parameters({'Action': 'CreateApplicationVersion', 'ContentType': 'JSON', 'Version': '2010-12-01', 'ApplicationName': 'application1', 'AutoCreateApplication': 'true', 'SourceBundle.S3Bucket': 'mybucket', 'SourceBundle.S3Key': 'mykey', 'VersionLabel': 'version1'})
        self.assertEqual(app_version['ApplicationName'], 'application1')
        self.assertEqual(app_version['VersionLabel'], 'version1')