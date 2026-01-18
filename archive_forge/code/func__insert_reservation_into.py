import abc
import datetime
from typing import Dict, List, Optional, overload, TYPE_CHECKING, Union
from google.protobuf.timestamp_pb2 import Timestamp
from cirq_google.engine import calibration
from cirq_google.cloud import quantum
from cirq_google.engine.abstract_processor import AbstractProcessor
from cirq_google.engine.abstract_program import AbstractProgram
def _insert_reservation_into(self, time_slot: quantum.QuantumTimeSlot) -> None:
    """Inserts a new reservation time slot into the ordered schedule.

        If this reservation overlaps with existing time slots, these slots will be
        shortened, removed, or split to insert the new reservation.
        """
    new_schedule = []
    time_slot_inserted = False
    for t in self._schedule:
        if t.end_time and t.end_time.timestamp() <= time_slot.start_time.timestamp():
            new_schedule.append(t)
            continue
        if t.start_time and t.start_time.timestamp() >= time_slot.end_time.timestamp():
            new_schedule.append(t)
            continue
        if t.start_time and time_slot.start_time.timestamp() <= t.start_time.timestamp():
            if not time_slot_inserted:
                new_schedule.append(time_slot)
                time_slot_inserted = True
            if not t.end_time or t.end_time.timestamp() > time_slot.end_time.timestamp():
                t.start_time = time_slot.end_time
                new_schedule.append(t)
        else:
            if not t.end_time or t.end_time.timestamp() > time_slot.end_time.timestamp():
                start = quantum.QuantumTimeSlot(processor_name=self._processor_id, end_time=time_slot.start_time, time_slot_type=t.time_slot_type)
                if t.start_time:
                    start.start_time = t.start_time
                end = quantum.QuantumTimeSlot(processor_name=self._processor_id, start_time=time_slot.end_time, time_slot_type=t.time_slot_type)
                if t.end_time:
                    end.end_time = t.end_time
                new_schedule.append(start)
                new_schedule.append(time_slot)
                new_schedule.append(end)
            else:
                t.end_time = time_slot.start_time
                new_schedule.append(t)
                new_schedule.append(time_slot)
            time_slot_inserted = True
    if not time_slot_inserted:
        new_schedule.append(time_slot)
    self._schedule = new_schedule