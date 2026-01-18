import collections
import os
import sys
import queue
import subprocess
import traceback
import weakref
from functools import partial
from threading import Thread
from jedi._compatibility import pickle_dump, pickle_load
from jedi import debug
from jedi.cache import memoize_method
from jedi.inference.compiled.subprocess import functions
from jedi.inference.compiled.access import DirectObjectAccess, AccessPath, \
from jedi.api.exceptions import InternalError
def _get_inference_state(self, function, inference_state_id):
    from jedi.inference import InferenceState
    try:
        inference_state = self._inference_states[inference_state_id]
    except KeyError:
        from jedi import InterpreterEnvironment
        inference_state = InferenceState(project=None, environment=InterpreterEnvironment())
        self._inference_states[inference_state_id] = inference_state
    return inference_state