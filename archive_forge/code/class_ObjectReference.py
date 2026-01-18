from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
class ObjectReference(_messages.Message):
    """ObjectReference contains enough information to let you inspect or modify
  the referred object.

  Fields:
    apiVersion: API version of the referent. +optional
    fieldPath: If referring to a piece of an object instead of an entire
      object, this string should contain a valid JSON/Go field access
      statement, such as desiredState.manifest.containers[2]. For example, if
      the object reference is to a container within a pod, this would take on
      a value like: "spec.containers{name}" (where "name" refers to the name
      of the container that triggered the event) or if no container name is
      specified "spec.containers[2]" (container with index 2 in this pod).
      This syntax is chosen only to have some well-defined way of referencing
      a part of an object.
    kind: Kind of the referent. More info:
      https://git.k8s.io/community/contributors/devel/api-
      conventions.md#types-kinds +optional
    name: Name of the referent. More info:
      https://kubernetes.io/docs/concepts/overview/working-with-
      objects/names/#names +optional
    namespace: Namespace of the referent. More info:
      https://kubernetes.io/docs/concepts/overview/working-with-
      objects/namespaces/ +optional
    resourceVersion: Specific resourceVersion to which this reference is made,
      if any. More info: https://git.k8s.io/community/contributors/devel/api-
      conventions.md#concurrency-control-and-consistency +optional
    uid: UID of the referent. More info:
      https://kubernetes.io/docs/concepts/overview/working-with-
      objects/names/#uids +optional
  """
    apiVersion = _messages.StringField(1)
    fieldPath = _messages.StringField(2)
    kind = _messages.StringField(3)
    name = _messages.StringField(4)
    namespace = _messages.StringField(5)
    resourceVersion = _messages.StringField(6)
    uid = _messages.StringField(7)