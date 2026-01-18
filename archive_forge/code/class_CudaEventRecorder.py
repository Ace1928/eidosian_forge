from collections import defaultdict, deque
from functools import partial
import statistics
from typing import ClassVar, Deque, Dict, Optional
import torch
class CudaEventRecorder(EventRecorder):
    """Allows profiling in an easy-to-use manner. CudaEventRecorder can be used
    in a loop. When it is used in a loop (or when an event recorder is created
    multiple times with the same name), get_timings returns the statistics of the
    timings since the last reset. Note: in case the number of timings is greater than
    10,000, only the last 10,000 timings are used to calculate the statistics.

    Usage:
    >>> event_recorder1 = CudaEventRecorder('1')
    >>> # Sequence of events whose time is to be measured
    >>> event_recorder1.stop()
    >>> event_recorder2 = CudaEventRecorder('2')
    >>> # Sequence of events whose time is to be measured
    >>> event_recorder2.stop()
    >>> print(CudaEventRecorder.get_timings())

    Args:
        event_name (str): The name by which the cuda event is to be referred later on

    """
    event_recorders: ClassVar[Dict[str, Deque['CudaEventRecorder']]] = defaultdict(deque_with_max_len_fixed)
    all_event_recorders: ClassVar[Dict[str, Deque['CudaEventRecorder']]] = defaultdict(deque_with_max_len_fixed)

    def __init__(self, event_name: str) -> None:
        self.event_name = event_name
        self.start_event = create_and_record_event()
        self.end_event: Optional[torch.cuda.Event] = None
        CudaEventRecorder.event_recorders[event_name].append(self)
        CudaEventRecorder.all_event_recorders[event_name].append(self)

    def stop(self) -> None:
        self.end_event = create_and_record_event()

    def find_time_elapsed(self) -> float:
        if self.end_event is None:
            raise Exception(f'stopEvent was not called for event with name {self.event_name}')
        self.end_event.synchronize()
        return self.start_event.elapsed_time(self.end_event)

    @classmethod
    def reset(cls) -> None:
        cls.event_recorders = defaultdict(deque_with_max_len_fixed)

    @classmethod
    def get_common_timings(cls, event_recorders: Dict[str, Deque['CudaEventRecorder']], description: str) -> str:
        all_timings_str = f'{description}:\n'
        for event_name, event_recorder_list in event_recorders.items():
            time_taken_list = [event_recorder.find_time_elapsed() for event_recorder in event_recorder_list]
            all_timings_str += '{}: Time taken: avg: {}, std: {}, count: {}\n'.format(event_name, statistics.mean(time_taken_list), statistics.pstdev(time_taken_list), len(time_taken_list))
        return all_timings_str

    @classmethod
    def get_timings(cls) -> str:
        """Returns the timings since last reset was called"""
        return cls.get_common_timings(cls.event_recorders, 'Timings since last reset')

    @classmethod
    def get_all_timings(cls) -> str:
        """Returns the statistics of all the timings"""
        return cls.get_common_timings(cls.all_event_recorders, 'All timings')