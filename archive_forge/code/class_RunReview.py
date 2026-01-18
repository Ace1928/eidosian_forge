import contextlib
import itertools
import logging
import os
import shutil
import socket
import sys
import tempfile
import threading
import time
from oslo_utils import timeutils
from oslo_utils import uuidutils
from zake import fake_client
from taskflow.conductors import backends as conductors
from taskflow import engines
from taskflow.jobs import backends as boards
from taskflow.patterns import linear_flow
from taskflow.persistence import backends as persistence
from taskflow.persistence import models
from taskflow import task
from taskflow.utils import threading_utils
class RunReview(task.Task):

    def _clone_review(self, review, temp_dir):
        print("Cloning review '%s' into %s" % (review['id'], temp_dir))

    def _run_tox(self, temp_dir):
        print('Running tox in %s' % temp_dir)

    def execute(self, review, temp_dir):
        self._clone_review(review, temp_dir)
        self._run_tox(temp_dir)