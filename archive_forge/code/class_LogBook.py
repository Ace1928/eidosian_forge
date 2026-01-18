import abc
import copy
import os
from oslo_utils import timeutils
from oslo_utils import uuidutils
from taskflow import exceptions as exc
from taskflow import states
from taskflow.types import failure as ft
from taskflow.utils import misc
class LogBook(object):
    """A collection of flow details and associated metadata.

    Typically this class contains a collection of flow detail entries
    for a given engine (or job) so that those entities can track what 'work'
    has been completed for resumption, reverting and miscellaneous tracking
    purposes.

    The data contained within this class need **not** be persisted to the
    backend storage in real time. The data in this class will only be
    guaranteed to be persisted when a save occurs via some backend
    connection.

    NOTE(harlowja): the naming of this class is analogous to a ship's log or a
    similar type of record used in detailing work that has been completed (or
    work that has not been completed).

    :ivar created_at: A ``datetime.datetime`` object of when this logbook
                      was created.
    :ivar updated_at: A ``datetime.datetime`` object of when this logbook
                      was last updated at.
    :ivar meta: A dictionary of meta-data associated with this logbook.
    """

    def __init__(self, name, uuid=None):
        if uuid:
            self._uuid = uuid
        else:
            self._uuid = uuidutils.generate_uuid()
        self._name = name
        self._flowdetails_by_id = {}
        self.created_at = timeutils.utcnow()
        self.updated_at = None
        self.meta = {}

    def pformat(self, indent=0, linesep=os.linesep):
        """Pretty formats this logbook into a string.

        >>> from taskflow.persistence import models
        >>> tmp = models.LogBook("example")
        >>> print(tmp.pformat())
        LogBook: 'example'
         - uuid = ...
         - created_at = ...
        """
        cls_name = self.__class__.__name__
        lines = ["%s%s: '%s'" % (' ' * indent, cls_name, self.name)]
        lines.extend(_format_shared(self, indent=indent + 1))
        lines.extend(_format_meta(self.meta, indent=indent + 1))
        if self.created_at is not None:
            lines.append('%s- created_at = %s' % (' ' * (indent + 1), self.created_at.isoformat()))
        if self.updated_at is not None:
            lines.append('%s- updated_at = %s' % (' ' * (indent + 1), self.updated_at.isoformat()))
        for flow_detail in self:
            lines.append(flow_detail.pformat(indent=indent + 1, linesep=linesep))
        return linesep.join(lines)

    def add(self, fd):
        """Adds a new flow detail into this logbook.

        NOTE(harlowja): if an existing flow detail exists with the same
        uuid the existing one will be overwritten with the newly provided
        one.

        Does not *guarantee* that the details will be immediately saved.
        """
        self._flowdetails_by_id[fd.uuid] = fd
        self.updated_at = timeutils.utcnow()

    def find(self, flow_uuid):
        """Locate the flow detail corresponding to the given uuid.

        :returns: the flow detail with that uuid
        :rtype: :py:class:`.FlowDetail` (or ``None`` if not found)
        """
        return self._flowdetails_by_id.get(flow_uuid, None)

    def merge(self, lb, deep_copy=False):
        """Merges the current object state with the given ones state.

        If ``deep_copy`` is provided as truthy then the
        local object will use ``copy.deepcopy`` to replace this objects
        local attributes with the provided objects attributes (**only** if
        there is a difference between this objects attributes and the
        provided attributes). If ``deep_copy`` is falsey (the default) then a
        reference copy will occur instead when a difference is detected.

        NOTE(harlowja): If the provided object is this object itself
        then **no** merging is done. Also note that this does **not** merge
        the flow details contained in either.

        :returns: this logbook (freshly merged with the incoming object)
        :rtype: :py:class:`.LogBook`
        """
        if lb is self:
            return self
        copy_fn = _copy_function(deep_copy)
        if self.meta != lb.meta:
            self.meta = copy_fn(lb.meta)
        if lb.created_at != self.created_at:
            self.created_at = copy_fn(lb.created_at)
        if lb.updated_at != self.updated_at:
            self.updated_at = copy_fn(lb.updated_at)
        return self

    def to_dict(self, marshal_time=False):
        """Translates the internal state of this object to a ``dict``.

        NOTE(harlowja): The returned ``dict`` does **not** include any
        contained flow details.

        :returns: this logbook in ``dict`` form
        """
        if not marshal_time:
            marshal_fn = lambda x: x
        else:
            marshal_fn = _safe_marshal_time
        return {'name': self.name, 'meta': self.meta, 'uuid': self.uuid, 'updated_at': marshal_fn(self.updated_at), 'created_at': marshal_fn(self.created_at)}

    @classmethod
    def from_dict(cls, data, unmarshal_time=False):
        """Translates the given ``dict`` into an instance of this class.

        NOTE(harlowja): the ``dict`` provided should come from a prior
        call to :meth:`.to_dict`.

        :returns: a new logbook
        :rtype: :py:class:`.LogBook`
        """
        if not unmarshal_time:
            unmarshal_fn = lambda x: x
        else:
            unmarshal_fn = _safe_unmarshal_time
        obj = cls(data['name'], uuid=data['uuid'])
        obj.updated_at = unmarshal_fn(data['updated_at'])
        obj.created_at = unmarshal_fn(data['created_at'])
        obj.meta = _fix_meta(data)
        return obj

    @property
    def uuid(self):
        """The unique identifer of this logbook."""
        return self._uuid

    @property
    def name(self):
        """The name of this logbook."""
        return self._name

    def __iter__(self):
        for fd in self._flowdetails_by_id.values():
            yield fd

    def __len__(self):
        return len(self._flowdetails_by_id)

    def copy(self, retain_contents=True):
        """Copies this logbook.

        Creates a shallow copy of this logbook. If this logbook contains
        flow details and ``retain_contents`` is truthy (the default) then
        the flow details container will be shallow copied (the flow details
        contained there-in will **not** be copied). If ``retain_contents`` is
        falsey then the copied logbook will have **no** contained flow
        details (but it will have the rest of the local objects attributes
        copied).

        :returns: a new logbook
        :rtype: :py:class:`.LogBook`
        """
        clone = copy.copy(self)
        if not retain_contents:
            clone._flowdetails_by_id = {}
        else:
            clone._flowdetails_by_id = self._flowdetails_by_id.copy()
        if self.meta:
            clone.meta = self.meta.copy()
        return clone