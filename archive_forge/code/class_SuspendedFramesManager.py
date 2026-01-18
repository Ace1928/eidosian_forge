from contextlib import contextmanager
import sys
from _pydevd_bundle.pydevd_constants import get_frame, RETURN_VALUES_DICT, \
from _pydevd_bundle.pydevd_xml import get_variable_details, get_type
from _pydev_bundle.pydev_override import overrides
from _pydevd_bundle.pydevd_resolver import sorted_attributes_key, TOO_LARGE_ATTR, get_var_scope
from _pydevd_bundle.pydevd_safe_repr import SafeRepr
from _pydev_bundle import pydev_log
from _pydevd_bundle import pydevd_vars
from _pydev_bundle.pydev_imports import Exec
from _pydevd_bundle.pydevd_frame_utils import FramesList
from _pydevd_bundle.pydevd_utils import ScopeRequest, DAPGrouper, Timer
from typing import Optional
class SuspendedFramesManager(object):

    def __init__(self):
        self._thread_id_to_fake_frames = {}
        self._thread_id_to_tracker = {}
        self._variable_reference_to_frames_tracker = {}

    def _get_tracker_for_variable_reference(self, variable_reference):
        tracker = self._variable_reference_to_frames_tracker.get(variable_reference)
        if tracker is not None:
            return tracker
        for _thread_id, tracker in self._thread_id_to_tracker.items():
            try:
                tracker.get_variable(variable_reference)
            except KeyError:
                pass
            else:
                return tracker
        return None

    def get_thread_id_for_variable_reference(self, variable_reference):
        """
        We can't evaluate variable references values on any thread, only in the suspended
        thread (the main reason for this is that in UI frameworks inspecting a UI object
        from a different thread can potentially crash the application).

        :param int variable_reference:
            The variable reference (can be either a frame id or a reference to a previously
            gotten variable).

        :return str:
            The thread id for the thread to be used to inspect the given variable reference or
            None if the thread was already resumed.
        """
        frames_tracker = self._get_tracker_for_variable_reference(variable_reference)
        if frames_tracker is not None:
            return frames_tracker.get_main_thread_id()
        return None

    def get_frame_tracker(self, thread_id):
        return self._thread_id_to_tracker.get(thread_id)

    def get_variable(self, variable_reference):
        """
        :raises KeyError
        """
        frames_tracker = self._get_tracker_for_variable_reference(variable_reference)
        if frames_tracker is None:
            raise KeyError()
        return frames_tracker.get_variable(variable_reference)

    def get_frames_list(self, thread_id):
        tracker = self._thread_id_to_tracker.get(thread_id)
        if tracker is None:
            return None
        return tracker.get_frames_list(thread_id)

    @contextmanager
    def track_frames(self, py_db):
        tracker = _FramesTracker(self, py_db)
        try:
            yield tracker
        finally:
            tracker.untrack_all()

    def add_fake_frame(self, thread_id, frame_id, frame):
        self._thread_id_to_fake_frames.setdefault(thread_id, {})[int(frame_id)] = frame

    def find_frame(self, thread_id, frame_id):
        try:
            if frame_id == '*':
                return get_frame()
            frame_id = int(frame_id)
            fake_frames = self._thread_id_to_fake_frames.get(thread_id)
            if fake_frames is not None:
                frame = fake_frames.get(frame_id)
                if frame is not None:
                    return frame
            frames_tracker = self._thread_id_to_tracker.get(thread_id)
            if frames_tracker is not None:
                frame = frames_tracker.find_frame(thread_id, frame_id)
                if frame is not None:
                    return frame
            return None
        except:
            pydev_log.exception()
            return None