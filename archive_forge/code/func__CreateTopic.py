from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import getopt
import re
import time
import uuid
from datetime import datetime
from gslib import metrics
from gslib.cloud_api import AccessDeniedException
from gslib.cloud_api import NotFoundException
from gslib.cloud_api import PublishPermissionDeniedException
from gslib.command import Command
from gslib.command import NO_MAX
from gslib.command_argument import CommandArgument
from gslib.cs_api_map import ApiSelector
from gslib.exception import CommandException
from gslib.help_provider import CreateHelpText
from gslib.project_id import PopulateProjectId
from gslib.pubsub_api import PubsubApi
from gslib.storage_url import StorageUrlFromString
from gslib.third_party.pubsub_apitools.pubsub_v1_messages import Binding
from gslib.utils import copy_helper
from gslib.utils import shim_util
from gslib.utils.shim_util import GcloudStorageFlag
from gslib.utils.shim_util import GcloudStorageMap
def _CreateTopic(self, pubsub_topic, service_account):
    """Assures that a topic exists, creating it if necessary.

    Also adds GCS as a publisher on that bucket, if necessary.

    Args:
      pubsub_topic: name of the Cloud Pub/Sub topic to use/create.
      service_account: the GCS service account that needs publish permission.

    Returns:
      true if we modified IAM permissions, otherwise false.
    """
    pubsub_api = PubsubApi(logger=self.logger)
    try:
        pubsub_api.GetTopic(topic_name=pubsub_topic)
        self.logger.debug('Topic %s already exists', pubsub_topic)
    except NotFoundException:
        self.logger.debug('Creating topic %s', pubsub_topic)
        pubsub_api.CreateTopic(topic_name=pubsub_topic)
        self.logger.info('Created Cloud Pub/Sub topic %s', pubsub_topic)
    policy = pubsub_api.GetTopicIamPolicy(topic_name=pubsub_topic)
    binding = Binding(role='roles/pubsub.publisher', members=['serviceAccount:%s' % service_account])
    if binding not in policy.bindings:
        policy.bindings.append(binding)
        pubsub_api.SetTopicIamPolicy(topic_name=pubsub_topic, policy=policy)
        return True
    else:
        self.logger.debug('GCS already has publish permission to topic %s.', pubsub_topic)
        return False