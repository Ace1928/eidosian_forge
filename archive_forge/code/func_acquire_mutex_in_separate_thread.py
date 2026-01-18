import threading
import uuid
from os_win import exceptions
from os_win.tests.functional import test_base
from os_win.utils import processutils
def acquire_mutex_in_separate_thread(self, mutex):
    stop_event = threading.Event()

    def target():
        mutex.acquire()
        stop_event.wait()
        mutex.release()
    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()
    return (thread, stop_event)