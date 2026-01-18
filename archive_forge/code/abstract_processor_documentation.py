import abc
import datetime
from typing import Dict, List, Optional, Sequence, TYPE_CHECKING, Union
import duet
import cirq
from cirq_google.api import v2
from cirq_google.cloud import quantum
from cirq_google.engine import calibration
Retrieves the schedule for a processor.

        The schedule may be filtered by time.

        Time slot type will be supported in the future.

        Args:
            from_time: Filters the returned schedule to only include entries
                that end no earlier than the given value. Specified either as an
                absolute time (datetime.datetime) or as a time relative to now
                (datetime.timedelta). Defaults to now (a relative time of 0).
                Set to None to omit this filter.
            to_time: Filters the returned schedule to only include entries
                that start no later than the given value. Specified either as an
                absolute time (datetime.datetime) or as a time relative to now
                (datetime.timedelta). Defaults to two weeks from now (a relative
                time of two weeks). Set to None to omit this filter.
            time_slot_type: Filters the returned schedule to only include
                entries with a given type (e.g. maintenance, open swim).
                Defaults to None. Set to None to omit this filter.

        Returns:
            Schedule time slots.
        