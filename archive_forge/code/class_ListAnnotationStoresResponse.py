from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ListAnnotationStoresResponse(_messages.Message):
    """Lists the Annotation stores in the given dataset.

  Fields:
    annotationStores: The returned Annotation stores. Won't be more Annotation
      stores than the value of page_size in the request.
    nextPageToken: Token to retrieve the next page of results or empty if
      there are no more results in the list.
  """
    annotationStores = _messages.MessageField('AnnotationStore', 1, repeated=True)
    nextPageToken = _messages.StringField(2)