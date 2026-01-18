import logging
import os
import sys
from concurrent import futures
import taskflow.engines
from taskflow.listeners import base
from taskflow.patterns import linear_flow as lf
from taskflow import states
from taskflow import task
from taskflow.types import notifier
class PokeFutureListener(base.Listener):

    def __init__(self, engine, future, task_name):
        super(PokeFutureListener, self).__init__(engine, task_listen_for=(notifier.Notifier.ANY,), flow_listen_for=[])
        self._future = future
        self._task_name = task_name

    def _task_receiver(self, state, details):
        if state in (states.SUCCESS, states.FAILURE):
            if details.get('task_name') == self._task_name:
                if state == states.SUCCESS:
                    self._future.set_result(details['result'])
                else:
                    failure = details['result']
                    self._future.set_exception(failure.exception)