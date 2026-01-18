import logging
import contextlib
import copy
import time
from asyncio import shield, Event, Future
from enum import Enum
from typing import Dict, FrozenSet, Iterable, List, Pattern, Set
from aiokafka.errors import IllegalStateError
from aiokafka.structs import OffsetAndMetadata, TopicPartition
from aiokafka.abc import ConsumerRebalanceListener
from aiokafka.util import create_future, get_running_loop
class TopicPartitionState:
    """ Shared Partition metadata state.

    After creation the workflow is similar to:

        * Partition assigned to this consumer (AWAITING_RESET)
        * Fetcher either uses commit save point or resets position in respect
          to defined reset policy (AWAITING_RESET -> CONSUMING)
        * Fetcher loads a new batch of records, yields results to consumer
          and updates consumed position (CONSUMING)
        * Assignment changed or subscription changed (CONSUMING -> UNASSIGNED)

    """

    def __init__(self, assignment):
        self._committed_futs = []
        self.highwater = None
        self.lso = None
        self.timestamp = None
        self._position = None
        self._position_fut = create_future()
        self._reset_strategy = None
        self._status = PartitionStatus.AWAITING_RESET
        self._assignment = assignment
        self._paused = False
        self._resume_fut = None

    @property
    def paused(self):
        return self._paused

    @property
    def resume_fut(self):
        return self._resume_fut

    @property
    def has_valid_position(self) -> bool:
        return self._position is not None

    @property
    def position(self) -> int:
        assert self._position is not None
        return self._position

    @property
    def awaiting_reset(self):
        return self._reset_strategy is not None

    @property
    def reset_strategy(self) -> int:
        return self._reset_strategy

    def await_reset(self, strategy):
        """ Called by either Coonsumer in `seek_to_*` or by Coordinator after
        setting initial committed point.
        """
        self._reset_strategy = strategy
        self._position = None
        if self._position_fut.done():
            self._position_fut = create_future()
        self._status = PartitionStatus.AWAITING_RESET

    def fetch_committed(self):
        fut = create_future()
        self._committed_futs.append(fut)
        self._assignment.commit_refresh_needed.set()
        return fut

    def update_committed(self, offset_meta: OffsetAndMetadata):
        """ Called by Coordinator on successful commit to update commit cache.
        """
        for fut in self._committed_futs:
            if not fut.done():
                fut.set_result(offset_meta)
        self._committed_futs.clear()

    def consumed_to(self, position: int):
        """ Called by Fetcher when yielding results to Consumer
        """
        assert self._status == PartitionStatus.CONSUMING
        self._position = position

    def reset_to(self, position: int):
        """ Called by Fetcher after performing a reset to force position to
        a new point.
        """
        assert self._status == PartitionStatus.AWAITING_RESET
        self._position = position
        self._reset_strategy = None
        self._status = PartitionStatus.CONSUMING
        if not self._position_fut.done():
            self._position_fut.set_result(None)

    def seek(self, position: int):
        """ Called by Consumer to force position to a specific offset
        """
        self._position = position
        self._reset_strategy = None
        self._status = PartitionStatus.CONSUMING
        if not self._position_fut.done():
            self._position_fut.set_result(None)

    def wait_for_position(self):
        return shield(self._position_fut)

    def pause(self):
        if not self._paused:
            self._paused = True
            assert self._resume_fut is None
            self._resume_fut = create_future()

    def resume(self):
        if self._paused:
            self._paused = False
            self._resume_fut.set_result(None)
            self._resume_fut = None

    def __repr__(self):
        return f'TopicPartitionState<Status={self._status} position={self._position}>'