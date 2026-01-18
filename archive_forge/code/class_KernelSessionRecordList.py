import os
import pathlib
import uuid
from typing import Any, Dict, List, NewType, Optional, Union, cast
from dataclasses import dataclass, fields
from jupyter_core.utils import ensure_async
from tornado import web
from traitlets import Instance, TraitError, Unicode, validate
from traitlets.config.configurable import LoggingConfigurable
from jupyter_server.traittypes import InstanceFromClasses
class KernelSessionRecordList:
    """An object for storing and managing a list of KernelSessionRecords.

    When adding a record to the list, the KernelSessionRecordList
    first checks if the record already exists in the list. If it does,
    the record will be updated with the new information; otherwise,
    it will be appended.
    """
    _records: List[KernelSessionRecord]

    def __init__(self, *records: KernelSessionRecord):
        """Initialize a record list."""
        self._records = []
        for record in records:
            self.update(record)

    def __str__(self):
        """The string representation of a record list."""
        return str(self._records)

    def __contains__(self, record: Union[KernelSessionRecord, str]) -> bool:
        """Search for records by kernel_id and session_id"""
        if isinstance(record, KernelSessionRecord) and record in self._records:
            return True
        if isinstance(record, str):
            for r in self._records:
                if record in [r.session_id, r.kernel_id]:
                    return True
        return False

    def __len__(self):
        """The length of the record list."""
        return len(self._records)

    def get(self, record: Union[KernelSessionRecord, str]) -> KernelSessionRecord:
        """Return a full KernelSessionRecord from a session_id, kernel_id, or
        incomplete KernelSessionRecord.
        """
        if isinstance(record, str):
            for r in self._records:
                if record in (r.kernel_id, r.session_id):
                    return r
        elif isinstance(record, KernelSessionRecord):
            for r in self._records:
                if record == r:
                    return record
        msg = f'{record} not found in KernelSessionRecordList.'
        raise ValueError(msg)

    def update(self, record: KernelSessionRecord) -> None:
        """Update a record in-place or append it if not in the list."""
        try:
            idx = self._records.index(record)
            self._records[idx].update(record)
        except ValueError:
            self._records.append(record)

    def remove(self, record: KernelSessionRecord) -> None:
        """Remove a record if its found in the list. If it's not found,
        do nothing.
        """
        if record in self._records:
            self._records.remove(record)