from itertools import chain
from Bio.SearchIO._utils import allitems, optionalcascade, getattr_str
from ._base import _BaseSearchObject
from .hsp import HSP
def _validate_hsp(self, hsp):
    """Validate an HSP object (PRIVATE).

        Valid HSP objects have the same hit_id as the Hit object ID and the
        same query_id as the Hit object's query_id.

        """
    if not isinstance(hsp, HSP):
        raise TypeError('Hit objects can only contain HSP objects.')
    if self._items:
        if self.id is not None:
            if hsp.hit_id != self.id:
                raise ValueError('Expected HSP with hit ID %r, found %r instead.' % (self.id, hsp.hit_id))
        else:
            self.id = hsp.hit_id
        if self.description is not None:
            if hsp.hit_description != self.description:
                raise ValueError('Expected HSP with hit description %r, found %r instead.' % (self.description, hsp.hit_description))
        else:
            self.description = hsp.hit_description
        if self.query_id is not None:
            if hsp.query_id != self.query_id:
                raise ValueError('Expected HSP with query ID %r, found %r instead.' % (self.query_id, hsp.query_id))
        else:
            self.query_id = hsp.query_id
        if self.query_description is not None:
            if hsp.query_description != self.query_description:
                raise ValueError('Expected HSP with query description %r, found %r instead.' % (self.query_description, hsp.query_description))
        else:
            self.query_description = hsp.query_description