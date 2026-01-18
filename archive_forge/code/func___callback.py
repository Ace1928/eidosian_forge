from pynvml.nvml import *
import datetime
import collections
import time
from threading import Thread
def __callback(self, result):
    self.__last_result = result
    if self.__callback_chain is not None:
        self.__callback_chain(self, result)