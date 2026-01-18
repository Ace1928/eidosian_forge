from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import functools
import time
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.storage import api_factory
from googlecloudsdk.api_lib.storage import cloud_api
from googlecloudsdk.api_lib.storage import errors as api_errors
from googlecloudsdk.api_lib.storage.gcs_json import error_util
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.storage import notification_configuration_iterator
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
@error_util.catch_http_error_raise_gcs_api_error()
def _maybe_create_or_modify_topic(topic_name, service_account_email):
    """Ensures that topic with SA permissions exists, creating it if needed.

  Args:
    topic_name (str): Name of the Cloud Pub/Sub topic to use or create.
    service_account_email (str): The project service account for Google Cloud
      Storage. This SA needs publish permission on the PubSub topic.

  Returns:
    True if topic was created or had its IAM permissions modified.
    Otherwise, False.
  """
    pubsub_client = apis.GetClientInstance('pubsub', 'v1')
    pubsub_messages = apis.GetMessagesModule('pubsub', 'v1')
    try:
        pubsub_client.projects_topics.Get(pubsub_messages.PubsubProjectsTopicsGetRequest(topic=topic_name))
        log.warning('Topic already exists: ' + topic_name)
        created_new_topic = False
    except apitools_exceptions.HttpError as e:
        if e.status_code != 404:
            raise
        new_topic = pubsub_client.projects_topics.Create(pubsub_messages.Topic(name=topic_name))
        log.info('Created topic:\n{}'.format(new_topic))
        created_new_topic = True
    topic_iam_policy = pubsub_client.projects_topics.GetIamPolicy(pubsub_messages.PubsubProjectsTopicsGetIamPolicyRequest(resource=topic_name))
    expected_binding = pubsub_messages.Binding(role='roles/pubsub.publisher', members=['serviceAccount:' + service_account_email])
    if expected_binding not in topic_iam_policy.bindings:
        topic_iam_policy.bindings.append(expected_binding)
        updated_topic_iam_policy = pubsub_client.projects_topics.SetIamPolicy(pubsub_messages.PubsubProjectsTopicsSetIamPolicyRequest(resource=topic_name, setIamPolicyRequest=pubsub_messages.SetIamPolicyRequest(policy=topic_iam_policy)))
        log.info('Updated topic IAM policy:\n{}'.format(updated_topic_iam_policy))
        return True
    else:
        log.warning('Project service account {} already has publish permission for topic {}'.format(service_account_email, topic_name))
    return created_new_topic