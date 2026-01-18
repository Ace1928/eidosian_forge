import gc
import pickle
import threading
import time
import weakref
from twisted._threads import Team, createMemoryWorker
from twisted.python import context, failure, threadable, threadpool
from twisted.trial import unittest
class Synchronization:
    failures = 0

    def __init__(self, N, waiting):
        self.N = N
        self.waiting = waiting
        self.lock = threading.Lock()
        self.runs = []

    def run(self):
        if self.lock.acquire(False):
            if not len(self.runs) % 5:
                time.sleep(0.0002)
            self.lock.release()
        else:
            self.failures += 1
        self.lock.acquire()
        self.runs.append(None)
        if len(self.runs) == self.N:
            self.waiting.release()
        self.lock.release()
    synchronized = ['run']