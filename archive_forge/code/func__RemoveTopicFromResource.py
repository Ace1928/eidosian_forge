from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def _RemoveTopicFromResource(topic_name, resource, resource_name):
    """Remove the topic from the Repo/ProjectConfig message."""
    if resource.pubsubConfigs is None:
        raise InvalidTopicError('Invalid topic [{0}]: No topics are configured in the {1}.'.format(topic_name, resource_name))
    config_additional_properties = resource.pubsubConfigs.additionalProperties
    for i, config in enumerate(config_additional_properties):
        if config.key == topic_name:
            del config_additional_properties[i]
            break
    else:
        raise InvalidTopicError('Invalid topic [{0}]: You must specify a topic that is already configured in the {1}.'.format(topic_name, resource_name))
    resource_msg_module = _MESSAGES.ProjectConfig
    if resource_name == 'repo':
        resource_msg_module = _MESSAGES.Repo
    return resource_msg_module(name=resource.name, pubsubConfigs=resource_msg_module.PubsubConfigsValue(additionalProperties=config_additional_properties))