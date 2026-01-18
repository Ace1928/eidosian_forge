import functools
import multiprocessing
import queue
import threading
import time
from threading import Event
from typing import Any, Callable, Dict, List, Optional
import psutil
import wandb
import wandb.util
from wandb.proto import wandb_internal_pb2 as pb
from wandb.sdk.internal.settings_static import SettingsStatic
from wandb.sdk.lib.mailbox import (
from wandb.sdk.lib.printer import get_printer
from wandb.sdk.wandb_run import Run
from ..interface.interface_relay import InterfaceRelay
def _finish_all(self, streams: Dict[str, StreamRecord], exit_code: int) -> None:
    if not streams:
        return
    printer = get_printer(all((stream._settings._jupyter for stream in streams.values())))
    self._printer = printer
    exit_handles = []
    started_streams: Dict[str, StreamRecord] = {}
    not_started_streams: Dict[str, StreamRecord] = {}
    for stream_id, stream in streams.items():
        d = started_streams if stream._started else not_started_streams
        d[stream_id] = stream
    for stream in started_streams.values():
        handle = stream.interface.deliver_exit(exit_code)
        handle.add_progress(self._on_progress_exit)
        handle.add_probe(functools.partial(self._on_probe_exit, stream=stream))
        exit_handles.append(handle)
    got_result = self._mailbox.wait_all(handles=exit_handles, timeout=-1, on_progress_all=self._on_progress_exit_all)
    assert got_result
    for _sid, stream in started_streams.items():
        poll_exit_handle = stream.interface.deliver_poll_exit()
        server_info_handle = stream.interface.deliver_request_server_info()
        final_summary_handle = stream.interface.deliver_get_summary()
        sampled_history_handle = stream.interface.deliver_request_sampled_history()
        internal_messages_handle = stream.interface.deliver_internal_messages()
        result = internal_messages_handle.wait(timeout=-1)
        assert result
        internal_messages_response = result.response.internal_messages_response
        job_info_handle = stream.interface.deliver_request_job_info()
        result = poll_exit_handle.wait(timeout=-1)
        assert result
        poll_exit_response = result.response.poll_exit_response
        result = server_info_handle.wait(timeout=-1)
        assert result
        server_info_response = result.response.server_info_response
        result = sampled_history_handle.wait(timeout=-1)
        assert result
        sampled_history = result.response.sampled_history_response
        result = final_summary_handle.wait(timeout=-1)
        assert result
        final_summary = result.response.get_summary_response
        result = job_info_handle.wait(timeout=-1)
        assert result
        job_info = result.response.job_info_response
        Run._footer(sampled_history=sampled_history, final_summary=final_summary, poll_exit_response=poll_exit_response, server_info_response=server_info_response, internal_messages_response=internal_messages_response, job_info=job_info, settings=stream._settings, printer=printer)
        stream.join()
    for stream in not_started_streams.values():
        stream.join()