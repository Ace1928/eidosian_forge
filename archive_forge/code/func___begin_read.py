from threading import Lock
def __begin_read(self):
    lock = self.__lock
    lock.acquire()
    self.__visit_count = self.__visit_count + 1
    lock.release()