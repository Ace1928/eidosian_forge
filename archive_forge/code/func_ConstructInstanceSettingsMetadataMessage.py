from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def ConstructInstanceSettingsMetadataMessage(message_classes, metadata):
    is_metadata = message_classes.InstanceSettingsMetadata().ItemsValue()
    if metadata.items():
        for key, value in metadata.items():
            is_metadata.additionalProperties.append(is_metadata.AdditionalProperty(key=key, value=value))
    return message_classes.InstanceSettingsMetadata(items=is_metadata)