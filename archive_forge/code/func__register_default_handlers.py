import copy
import os
import botocore.session
from botocore.client import Config
from botocore.exceptions import DataNotFoundError, UnknownServiceError
import boto3
import boto3.utils
from boto3.exceptions import ResourceNotExistsError, UnknownAPIVersionError
from .resources.factory import ResourceFactory
def _register_default_handlers(self):
    self._session.register('creating-client-class.s3', boto3.utils.lazy_call('boto3.s3.inject.inject_s3_transfer_methods'))
    self._session.register('creating-resource-class.s3.Bucket', boto3.utils.lazy_call('boto3.s3.inject.inject_bucket_methods'))
    self._session.register('creating-resource-class.s3.Object', boto3.utils.lazy_call('boto3.s3.inject.inject_object_methods'))
    self._session.register('creating-resource-class.s3.ObjectSummary', boto3.utils.lazy_call('boto3.s3.inject.inject_object_summary_methods'))
    self._session.register('creating-resource-class.dynamodb', boto3.utils.lazy_call('boto3.dynamodb.transform.register_high_level_interface'), unique_id='high-level-dynamodb')
    self._session.register('creating-resource-class.dynamodb.Table', boto3.utils.lazy_call('boto3.dynamodb.table.register_table_methods'), unique_id='high-level-dynamodb-table')
    self._session.register('creating-resource-class.ec2.ServiceResource', boto3.utils.lazy_call('boto3.ec2.createtags.inject_create_tags'))
    self._session.register('creating-resource-class.ec2.Instance', boto3.utils.lazy_call('boto3.ec2.deletetags.inject_delete_tags', event_emitter=self.events))