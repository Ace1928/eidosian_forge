from __future__ import annotations
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import cast
from streamlit import util
from streamlit.proto.WidgetStates_pb2 import WidgetStates
from streamlit.runtime.state import coalesce_widget_states
class ScriptRequests:
    """An interface for communicating with a ScriptRunner. Thread-safe.

    AppSession makes requests of a ScriptRunner through this class, and
    ScriptRunner handles those requests.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._state = ScriptRequestType.CONTINUE
        self._rerun_data = RerunData()

    def request_stop(self) -> None:
        """Request that the ScriptRunner stop running. A stopped ScriptRunner
        can't be used anymore. STOP requests succeed unconditionally.
        """
        with self._lock:
            self._state = ScriptRequestType.STOP

    def request_rerun(self, new_data: RerunData) -> bool:
        """Request that the ScriptRunner rerun its script.

        If the ScriptRunner has been stopped, this request can't be honored:
        return False.

        Otherwise, record the request and return True. The ScriptRunner will
        handle the rerun request as soon as it reaches an interrupt point.
        """
        with self._lock:
            if self._state == ScriptRequestType.STOP:
                return False
            if self._state == ScriptRequestType.CONTINUE:
                self._state = ScriptRequestType.RERUN
                self._rerun_data = new_data
                return True
            if self._state == ScriptRequestType.RERUN:
                coalesced_states = coalesce_widget_states(self._rerun_data.widget_states, new_data.widget_states)
                if new_data.fragment_id_queue:
                    fragment_id_queue = [*self._rerun_data.fragment_id_queue]
                    if (new_fragment_id := new_data.fragment_id_queue[0]) not in fragment_id_queue:
                        fragment_id_queue.append(new_fragment_id)
                else:
                    fragment_id_queue = []
                self._rerun_data = RerunData(query_string=new_data.query_string, widget_states=coalesced_states, page_script_hash=new_data.page_script_hash, page_name=new_data.page_name, fragment_id_queue=fragment_id_queue)
                return True
            raise RuntimeError(f'Unrecognized ScriptRunnerState: {self._state}')

    def on_scriptrunner_yield(self) -> ScriptRequest | None:
        """Called by the ScriptRunner when it's at a yield point.

        If we have no request or a RERUN request corresponding to one or more fragments,
        return None.

        If we have a (full script) RERUN request, return the request and set our internal
        state to CONTINUE.

        If we have a STOP request, return the request and remain stopped.
        """
        if self._state == ScriptRequestType.CONTINUE or (self._state == ScriptRequestType.RERUN and self._rerun_data.fragment_id_queue):
            return None
        with self._lock:
            if self._state == ScriptRequestType.RERUN:
                if self._rerun_data.fragment_id_queue:
                    return None
                self._state = ScriptRequestType.CONTINUE
                return ScriptRequest(ScriptRequestType.RERUN, self._rerun_data)
            assert self._state == ScriptRequestType.STOP
            return ScriptRequest(ScriptRequestType.STOP)

    def on_scriptrunner_ready(self) -> ScriptRequest:
        """Called by the ScriptRunner when it's about to run its script for
        the first time, and also after its script has successfully completed.

        If we have a RERUN request, return the request and set
        our internal state to CONTINUE.

        If we have a STOP request or no request, set our internal state
        to STOP.
        """
        with self._lock:
            if self._state == ScriptRequestType.RERUN:
                self._state = ScriptRequestType.CONTINUE
                return ScriptRequest(ScriptRequestType.RERUN, self._rerun_data)
            self._state = ScriptRequestType.STOP
            return ScriptRequest(ScriptRequestType.STOP)