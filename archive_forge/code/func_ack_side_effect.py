from taskflow.engines.worker_based import dispatcher
from taskflow import test
from taskflow.test import mock
def ack_side_effect(*args, **kwargs):
    msg.acknowledged = True