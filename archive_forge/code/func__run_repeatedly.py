import threading
from tensorboard import errors
def _run_repeatedly():
    while True:
        target()
        event.wait(interval_sec)
        event.clear()