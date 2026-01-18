import abc
import copy
import os
from oslo_utils import timeutils
from oslo_utils import uuidutils
from taskflow import exceptions as exc
from taskflow import states
from taskflow.types import failure as ft
from taskflow.utils import misc
class FlowDetail(object):
    """A collection of atom details and associated metadata.

    Typically this class contains a collection of atom detail entries that
    represent the atoms in a given flow structure (along with any other needed
    metadata relevant to that flow).

    The data contained within this class need **not** be persisted to the
    backend storage in real time. The data in this class will only be
    guaranteed to be persisted when a save (or update) occurs via some backend
    connection.

    :ivar meta: A dictionary of meta-data associated with this flow detail.
    """

    def __init__(self, name, uuid):
        self._uuid = uuid
        self._name = name
        self._atomdetails_by_id = {}
        self.state = None
        self.meta = {}

    def update(self, fd):
        """Updates the objects state to be the same as the given one.

        This will assign the private and public attributes of the given
        flow detail directly to this object (replacing any existing
        attributes in this object; even if they are the **same**).

        NOTE(harlowja): If the provided object is this object itself
        then **no** update is done.

        :returns: this flow detail
        :rtype: :py:class:`.FlowDetail`
        """
        if fd is self:
            return self
        self._atomdetails_by_id = fd._atomdetails_by_id
        self.state = fd.state
        self.meta = fd.meta
        return self

    def pformat(self, indent=0, linesep=os.linesep):
        """Pretty formats this flow detail into a string.

        >>> from oslo_utils import uuidutils
        >>> from taskflow.persistence import models
        >>> flow_detail = models.FlowDetail("example",
        ...                                 uuid=uuidutils.generate_uuid())
        >>> print(flow_detail.pformat())
        FlowDetail: 'example'
         - uuid = ...
         - state = ...
        """
        cls_name = self.__class__.__name__
        lines = ["%s%s: '%s'" % (' ' * indent, cls_name, self.name)]
        lines.extend(_format_shared(self, indent=indent + 1))
        lines.extend(_format_meta(self.meta, indent=indent + 1))
        for atom_detail in self:
            lines.append(atom_detail.pformat(indent=indent + 1, linesep=linesep))
        return linesep.join(lines)

    def merge(self, fd, deep_copy=False):
        """Merges the current object state with the given one's state.

        If ``deep_copy`` is provided as truthy then the
        local object will use ``copy.deepcopy`` to replace this objects
        local attributes with the provided objects attributes (**only** if
        there is a difference between this objects attributes and the
        provided attributes). If ``deep_copy`` is falsey (the default) then a
        reference copy will occur instead when a difference is detected.

        NOTE(harlowja): If the provided object is this object itself
        then **no** merging is done. Also this does **not** merge the atom
        details contained in either.

        :returns: this flow detail (freshly merged with the incoming object)
        :rtype: :py:class:`.FlowDetail`
        """
        if fd is self:
            return self
        copy_fn = _copy_function(deep_copy)
        if self.meta != fd.meta:
            self.meta = copy_fn(fd.meta)
        if self.state != fd.state:
            self.state = fd.state
        return self

    def copy(self, retain_contents=True):
        """Copies this flow detail.

        Creates a shallow copy of this flow detail. If this detail contains
        flow details and ``retain_contents`` is truthy (the default) then
        the atom details container will be shallow copied (the atom details
        contained there-in will **not** be copied). If ``retain_contents`` is
        falsey then the copied flow detail will have **no** contained atom
        details (but it will have the rest of the local objects attributes
        copied).

        :returns: a new flow detail
        :rtype: :py:class:`.FlowDetail`
        """
        clone = copy.copy(self)
        if not retain_contents:
            clone._atomdetails_by_id = {}
        else:
            clone._atomdetails_by_id = self._atomdetails_by_id.copy()
        if self.meta:
            clone.meta = self.meta.copy()
        return clone

    def to_dict(self):
        """Translates the internal state of this object to a ``dict``.

        NOTE(harlowja): The returned ``dict`` does **not** include any
        contained atom details.

        :returns: this flow detail in ``dict`` form
        """
        return {'name': self.name, 'meta': self.meta, 'state': self.state, 'uuid': self.uuid}

    @classmethod
    def from_dict(cls, data):
        """Translates the given ``dict`` into an instance of this class.

        NOTE(harlowja): the ``dict`` provided should come from a prior
        call to :meth:`.to_dict`.

        :returns: a new flow detail
        :rtype: :py:class:`.FlowDetail`
        """
        obj = cls(data['name'], data['uuid'])
        obj.state = data.get('state')
        obj.meta = _fix_meta(data)
        return obj

    def add(self, ad):
        """Adds a new atom detail into this flow detail.

        NOTE(harlowja): if an existing atom detail exists with the same
        uuid the existing one will be overwritten with the newly provided
        one.

        Does not *guarantee* that the details will be immediately saved.
        """
        self._atomdetails_by_id[ad.uuid] = ad

    def find(self, ad_uuid):
        """Locate the atom detail corresponding to the given uuid.

        :returns: the atom detail with that uuid
        :rtype: :py:class:`.AtomDetail` (or ``None`` if not found)
        """
        return self._atomdetails_by_id.get(ad_uuid)

    @property
    def uuid(self):
        """The unique identifer of this flow detail."""
        return self._uuid

    @property
    def name(self):
        """The name of this flow detail."""
        return self._name

    def __iter__(self):
        for ad in self._atomdetails_by_id.values():
            yield ad

    def __len__(self):
        return len(self._atomdetails_by_id)