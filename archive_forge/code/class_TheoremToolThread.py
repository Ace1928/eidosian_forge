import threading
import time
from abc import ABCMeta, abstractmethod
class TheoremToolThread(threading.Thread):

    def __init__(self, command, verbose, name=None):
        threading.Thread.__init__(self)
        self._command = command
        self._result = None
        self._verbose = verbose
        self._name = name

    def run(self):
        try:
            self._result = self._command()
            if self._verbose:
                print('Thread %s finished with result %s at %s' % (self._name, self._result, time.localtime(time.time())))
        except Exception as e:
            print(e)
            print('Thread %s completed abnormally' % self._name)

    @property
    def result(self):
        return self._result