import time
from boto.elastictranscoder.layer1 import ElasticTranscoderConnection
from boto.elastictranscoder.exceptions import ValidationException
from tests.compat import unittest
import boto.s3
import boto.sns
import boto.iam
import boto.sns
class TestETSLayer1PipelineManagement(unittest.TestCase):

    def setUp(self):
        self.api = ElasticTranscoderConnection()
        self.s3 = boto.connect_s3()
        self.sns = boto.connect_sns()
        self.iam = boto.connect_iam()
        self.sns = boto.connect_sns()
        self.timestamp = str(int(time.time()))
        self.input_bucket = 'boto-pipeline-%s' % self.timestamp
        self.output_bucket = 'boto-pipeline-out-%s' % self.timestamp
        self.role_name = 'boto-ets-role-%s' % self.timestamp
        self.pipeline_name = 'boto-pipeline-%s' % self.timestamp
        self.s3.create_bucket(self.input_bucket)
        self.s3.create_bucket(self.output_bucket)
        self.addCleanup(self.s3.delete_bucket, self.input_bucket)
        self.addCleanup(self.s3.delete_bucket, self.output_bucket)
        self.role = self.iam.create_role(self.role_name)
        self.role_arn = self.role['create_role_response']['create_role_result']['role']['arn']
        self.addCleanup(self.iam.delete_role, self.role_name)

    def create_pipeline(self):
        pipeline = self.api.create_pipeline(self.pipeline_name, self.input_bucket, self.output_bucket, self.role_arn, {'Progressing': '', 'Completed': '', 'Warning': '', 'Error': ''})
        pipeline_id = pipeline['Pipeline']['Id']
        self.addCleanup(self.api.delete_pipeline, pipeline_id)
        return pipeline_id

    def test_create_delete_pipeline(self):
        pipeline = self.api.create_pipeline(self.pipeline_name, self.input_bucket, self.output_bucket, self.role_arn, {'Progressing': '', 'Completed': '', 'Warning': '', 'Error': ''})
        pipeline_id = pipeline['Pipeline']['Id']
        self.api.delete_pipeline(pipeline_id)

    def test_can_retrieve_pipeline_information(self):
        pipeline_id = self.create_pipeline()
        pipelines = self.api.list_pipelines()['Pipelines']
        pipeline_names = [p['Name'] for p in pipelines]
        self.assertIn(self.pipeline_name, pipeline_names)
        response = self.api.read_pipeline(pipeline_id)
        self.assertEqual(response['Pipeline']['Id'], pipeline_id)

    def test_update_pipeline(self):
        pipeline_id = self.create_pipeline()
        self.api.update_pipeline_status(pipeline_id, 'Paused')
        response = self.api.read_pipeline(pipeline_id)
        self.assertEqual(response['Pipeline']['Status'], 'Paused')

    def test_update_pipeline_notification(self):
        pipeline_id = self.create_pipeline()
        response = self.sns.create_topic('pipeline-errors')
        topic_arn = response['CreateTopicResponse']['CreateTopicResult']['TopicArn']
        self.addCleanup(self.sns.delete_topic, topic_arn)
        self.api.update_pipeline_notifications(pipeline_id, {'Progressing': '', 'Completed': '', 'Warning': '', 'Error': topic_arn})
        response = self.api.read_pipeline(pipeline_id)
        self.assertEqual(response['Pipeline']['Notifications']['Error'], topic_arn)

    def test_list_jobs_by_pipeline(self):
        pipeline_id = self.create_pipeline()
        response = self.api.list_jobs_by_pipeline(pipeline_id)
        self.assertEqual(response['Jobs'], [])

    def test_proper_error_when_pipeline_does_not_exist(self):
        with self.assertRaises(ValidationException):
            self.api.read_pipeline('badpipelineid')