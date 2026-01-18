from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def _UpdateTopicInResource(topic_name, args, resource, resource_name):
    """Update the topic in the configuration and return a new repo message."""
    if resource.pubsubConfigs is None:
        raise InvalidTopicError('Invalid topic [{0}]: No topics are configured in the {1}.'.format(topic_name, resource_name))
    config_additional_properties = resource.pubsubConfigs.additionalProperties
    for i, config in enumerate(config_additional_properties):
        if config.key == topic_name:
            config_additional_properties[i].value = _UpdateConfigWithArgs(config.value, args)
            break
    else:
        raise InvalidTopicError('Invalid topic [{0}]: You must specify a topic that is already configured in the {1}.'.format(topic_name, resource_name))
    resource_msg_module = _MESSAGES.ProjectConfig
    if resource_name == 'repo':
        resource_msg_module = _MESSAGES.Repo
    return resource_msg_module(name=resource.name, pubsubConfigs=resource_msg_module.PubsubConfigsValue(additionalProperties=config_additional_properties))