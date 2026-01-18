from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import logging
import unittest
from gslib.cs_api_map import ApiSelector
from gslib.project_id import PopulateProjectId
from gslib.pubsub_api import PubsubApi
import gslib.tests.testcase as testcase
from gslib.tests.testcase.integration_testcase import SkipForS3
from gslib.tests.util import ObjectToURI as suri
def _RegisterDefaultTopicCreation(self, bucket_name):
    """Records the name of a topic we expect to create, for cleanup."""
    if self.test_api == ApiSelector.XML:
        return unittest.skip('Notifications only work with the JSON API.')
    expected_topic_name = 'projects/%s/topics/%s' % (PopulateProjectId(None), bucket_name)
    self.created_topic = expected_topic_name
    return expected_topic_name