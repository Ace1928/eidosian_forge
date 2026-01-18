import collections
import enum
import functools
import uuid
import ovs.db.data as data
import ovs.db.parser
import ovs.db.schema
import ovs.jsonrpc
import ovs.ovsuuid
import ovs.poller
import ovs.vlog
from ovs.db import custom_index
from ovs.db import error
class ConditionState(object):

    def __init__(self):
        self._ack_cond = [True]
        self._req_cond = None
        self._new_cond = None

    def __iter__(self):
        return iter([self._new_cond, self._req_cond, self._ack_cond])

    @property
    def new(self):
        """The latest freshly initialized condition change"""
        return self._new_cond

    @property
    def acked(self):
        """The last condition change that has been accepted by the server"""
        return self._ack_cond

    @property
    def requested(self):
        """A condition that's been requested, but not acked by the server"""
        return self._req_cond

    @property
    def latest(self):
        """The most recent condition change"""
        return next((cond for cond in self if cond is not None))

    @staticmethod
    def is_true(condition):
        return condition == [True]

    def init(self, cond):
        """Signal that a condition change is being initiated"""
        self._new_cond = cond

    def ack(self):
        """Signal that a condition change has been acked"""
        if self._req_cond is not None:
            self._ack_cond, self._req_cond = (self._req_cond, None)

    def request(self):
        """Signal that a condition change has been requested"""
        if self._new_cond is not None:
            self._req_cond, self._new_cond = (self._new_cond, None)

    def reset(self):
        """Reset a requested condition change back to new"""
        if self._req_cond is not None:
            if self._new_cond is None:
                self._new_cond = self._req_cond
            self._req_cond = None
            return True
        return False