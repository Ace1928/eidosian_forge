from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import abc
import collections
from apitools.base.protorpclite import messages
from googlecloudsdk.api_lib.run import condition
from googlecloudsdk.core.console import console_attr
import six
@six.add_metaclass(abc.ABCMeta)
class KubernetesObject(object):
    """Base class for wrappers around Kubernetes-style Object messages.

  Requires subclasses to provide class-level constants KIND for the k8s Kind
  field, and API_CATEGORY for the k8s API Category. It infers the API version
  from the version of the client object.

  Additionally, you can set READY_CONDITION and TERMINAL_CONDITIONS to be the
  name of a condition that indicates readiness, and a set of conditions
  indicating a steady state, respectively.
  """
    READY_CONDITION = 'Ready'

    @classmethod
    def Kind(cls, kind=None):
        """Returns the passed str if given, else the class KIND."""
        return kind if kind is not None else cls.KIND

    @classmethod
    def ApiCategory(cls, api_category=None):
        """Returns the passed str if given, else the class API_CATEGORY."""
        return api_category if api_category is not None else cls.API_CATEGORY

    @classmethod
    def ApiVersion(cls, api_version, api_category=None):
        """Returns the api version with group prefix if exists."""
        if api_category is None:
            return api_version
        return '{}/{}'.format(api_category, api_version)

    @classmethod
    def SpecOnly(cls, spec, messages_mod, kind=None):
        """Produces a wrapped message with only the given spec.

    It is meant to be used as part of another message; it will error if you
    try to access the metadata or status.

    Arguments:
      spec: messages.Message, The spec to include
      messages_mod: the messages module
      kind: str, the resource kind

    Returns:
      A new k8s_object with only the given spec.
    """
        msg_cls = getattr(messages_mod, cls.Kind(kind))
        return cls(msg_cls(spec=spec), messages_mod, kind)

    @classmethod
    def Template(cls, template, messages_mod, kind=None):
        """Wraps a template object: spec and metadata only, no status."""
        msg_cls = getattr(messages_mod, cls.Kind(kind))
        return cls(msg_cls(spec=template.spec, metadata=template.metadata), messages_mod, kind)

    @classmethod
    def New(cls, client, namespace, kind=None, api_category=None):
        """Produces a new wrapped message of the appropriate type.

    All the sub-objects in it are recursively initialized to the appropriate
    message types, and the kind, apiVersion, and namespace set.

    Arguments:
      client: the API client to use
      namespace: str, The namespace to create the object in
      kind: str, the resource kind
      api_category: str, the api group of the resource

    Returns:
      The newly created wrapped message.
    """
        api_category = cls.ApiCategory(api_category)
        api_version = cls.ApiVersion(getattr(client, '_VERSION'), api_category)
        messages_mod = client.MESSAGES_MODULE
        kind = cls.Kind(kind)
        ret = InitializedInstance(getattr(messages_mod, kind))
        try:
            ret.kind = kind
            ret.apiVersion = api_version
        except AttributeError:
            pass
        ret.metadata.namespace = namespace
        return cls(ret, messages_mod, kind)

    def __init__(self, to_wrap, messages_mod, kind=None):
        msg_cls = getattr(messages_mod, self.Kind(kind))
        if not isinstance(to_wrap, msg_cls):
            raise ValueError('Oops, trying to wrap wrong kind of message')
        self._m = to_wrap
        self._messages = messages_mod

    def MessagesModule(self):
        """Return the messages module."""
        return self._messages

    def AssertFullObject(self):
        if not self._m.metadata:
            raise ValueError('This instance is spec-only.')

    def IsFullObject(self):
        return self._m.metadata

    @property
    def kind(self):
        self.AssertFullObject()
        return self._m.kind

    @property
    def apiVersion(self):
        self.AssertFullObject()
        return self._m.apiVersion

    @property
    def spec(self):
        return self._m.spec

    @property
    def status(self):
        self.AssertFullObject()
        return self._m.status

    @property
    def metadata(self):
        self.AssertFullObject()
        return self._m.metadata

    @metadata.setter
    def metadata(self, value):
        self._m.metadata = value

    @property
    def name(self):
        self.AssertFullObject()
        return self._m.metadata.name

    @name.setter
    def name(self, value):
        self.AssertFullObject()
        self._m.metadata.name = value

    @property
    def author(self):
        return self.annotations.get(AUTHOR_ANNOTATION)

    @property
    def creation_timestamp(self):
        return self.metadata.creationTimestamp

    @property
    def namespace(self):
        self.AssertFullObject()
        return self._m.metadata.namespace

    @namespace.setter
    def namespace(self, value):
        self.AssertFullObject()
        self._m.metadata.namespace = value

    @property
    def resource_version(self):
        self.AssertFullObject()
        return self._m.metadata.resourceVersion

    @property
    def self_link(self):
        self.AssertFullObject()
        return self._m.metadata.selfLink.lstrip('/')

    @property
    def uid(self):
        self.AssertFullObject()
        return self._m.metadata.uid

    @property
    def owners(self):
        self.AssertFullObject()
        return self._m.metadata.ownerReferences

    @property
    def is_managed(self):
        return REGION_LABEL in self.labels

    @property
    def region(self):
        self.AssertFullObject()
        return self.labels[REGION_LABEL]

    @property
    def generation(self):
        self.AssertFullObject()
        return self._m.metadata.generation

    @generation.setter
    def generation(self, value):
        self._m.metadata.generation = value

    @property
    def conditions(self):
        return self.GetConditions()

    def GetConditions(self, terminal_condition=None):
        self.AssertFullObject()
        if self._m.status:
            c = self._m.status.conditions
        else:
            c = []
        return condition.Conditions(c, terminal_condition if terminal_condition else self.READY_CONDITION, getattr(self._m.status, 'observedGeneration', None), self.generation)

    @property
    def annotations(self):
        self.AssertFullObject()
        return AnnotationsFromMetadata(self._messages, self._m.metadata)

    @property
    def labels(self):
        self.AssertFullObject()
        return LabelsFromMetadata(self._messages, self._m.metadata)

    @property
    def ready_condition(self):
        assert hasattr(self, 'READY_CONDITION')
        if self.conditions and self.READY_CONDITION in self.conditions:
            return self.conditions[self.READY_CONDITION]

    @property
    def ready(self):
        assert hasattr(self, 'READY_CONDITION')
        if self.ready_condition:
            return self.ready_condition['status']

    @property
    def last_transition_time(self):
        assert hasattr(self, 'READY_CONDITION')
        if self.ready_condition:
            return self.ready_condition['lastTransitionTime']

    def _PickSymbol(self, best, alt, encoding):
        """Choose the best symbol (if it's in this encoding) or an alternate."""
        try:
            best.encode(encoding)
            return best
        except UnicodeError:
            return alt

    @property
    def ready_symbol(self):
        """Return a symbol summarizing the status of this object."""
        return self.ReadySymbolAndColor()[0]

    def ReadySymbolAndColor(self):
        """Return a tuple of ready_symbol and display color for this object."""
        encoding = console_attr.GetConsoleAttr().GetEncoding()
        if self.ready is None:
            return (self._PickSymbol('…', '.', encoding), 'yellow')
        elif self.ready:
            return (self._PickSymbol('✔', '+', encoding), 'green')
        else:
            return ('X', 'red')

    def AsObjectReference(self):
        return self._messages.ObjectReference(kind=self.kind, namespace=self.namespace, name=self.name, uid=self.uid, apiVersion=self.apiVersion)

    def Message(self):
        """Return the actual message we've wrapped."""
        return self._m

    def MakeSerializable(self):
        return self.Message()

    def MakeCondition(self, *args, **kwargs):
        if hasattr(self._messages, 'GoogleCloudRunV1Condition'):
            return self._messages.GoogleCloudRunV1Condition(*args, **kwargs)
        else:
            return getattr(self._messages, self.kind + 'Condition')(*args, **kwargs)

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return self.Message() == other.Message()
        return False

    def __repr__(self):
        return '{}({})'.format(type(self).__name__, repr(self._m))