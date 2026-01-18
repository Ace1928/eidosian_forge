from enum import Enum
from queue import Queue
from threading import Thread
from typing import Callable, Optional, List
from .errors import AsyncTaskException
class GlobalTaskManager:
    """
    Singleton class designed to manage async tasks in a thread pool.
    Multiple instances of this class can cause unintended behaviour
    """
    max_threads: Optional[int] = 32

    def __init__(self, max_threads: Optional[int]=None):
        if max_threads is not None:
            self.max_threads = 200 if max_threads <= 0 else max_threads
        self.thread_pool: List[Thread] = []
        self.task_queue = Queue()

    def put(self, thread: Thread):
        self.task_queue.put(thread)

    def remove(self, thread: Thread):
        for t in self.thread_pool:
            if thread.ident == t.ident:
                self.thread_pool.remove(t)

    def run(self):
        while True:
            if len(self.thread_pool) < self.max_threads:
                child_thread: Thread = self.task_queue.get()
                child_thread.start()
                self.thread_pool.append(child_thread)