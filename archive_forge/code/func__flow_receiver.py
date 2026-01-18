import logging
import os
from taskflow import exceptions
from taskflow.listeners import base
from taskflow import states
def _flow_receiver(self, state, details):
    self._claim_checker(state, details)