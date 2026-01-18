from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class APIVersionKindSelector(_messages.Message):
    """APIVersionKindSelector is an APIVersion Kind tuple with a LabelSelector.

  Fields:
    apiVersion: APIVersion - the API version of the resource to watch.
    kind: Kind of the resource to watch. More info:
      https://git.k8s.io/community/contributors/devel/sig-architecture/api-
      conventions.md#types-kinds
    selector: LabelSelector filters this source to objects to those resources
      pass the label selector. More info:
      http://kubernetes.io/docs/concepts/overview/working-with-
      objects/labels/#label-selectors
  """
    apiVersion = _messages.StringField(1)
    kind = _messages.StringField(2)
    selector = _messages.MessageField('LabelSelector', 3)