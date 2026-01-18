from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import json
import logging
import traceback
from apitools.base.py import exceptions as apitools_exceptions
from boto import config
from gslib.cloud_api import AccessDeniedException
from gslib.cloud_api import BadRequestException
from gslib.cloud_api import NotFoundException
from gslib.cloud_api import PreconditionException
from gslib.cloud_api import ServiceException
from gslib.gcs_json_credentials import SetUpJsonCredentialsAndCache
from gslib.no_op_credentials import NoOpCredentials
from gslib.third_party.pubsub_apitools import pubsub_v1_client as apitools_client
from gslib.third_party.pubsub_apitools import pubsub_v1_messages as apitools_messages
from gslib.utils import system_util
from gslib.utils.boto_util import GetCertsFile
from gslib.utils.boto_util import GetMaxRetryDelay
from gslib.utils.boto_util import GetNewHttp
from gslib.utils.boto_util import GetNumRetries
from gslib.utils.constants import UTF8
def DeleteTopic(self, topic_name):
    """Only used in system tests."""
    request = apitools_messages.PubsubProjectsTopicsDeleteRequest(topic=topic_name)
    try:
        return self.api_client.projects_topics.Delete(request)
    except TRANSLATABLE_APITOOLS_EXCEPTIONS as e:
        self._TranslateExceptionAndRaise(e, topic_name=topic_name)