from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
def VersionedInsertAnnotationConfig(annotation_store):
    if not annotation_store:
        return None
    messages = apis.GetMessagesModule('healthcare', api_version)
    return messages.AnnotationConfig(annotationStoreName=annotation_store, storeQuote=True)