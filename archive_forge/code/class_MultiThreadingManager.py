import sys
from concurrent.futures import ThreadPoolExecutor
from queue import PriorityQueue
class MultiThreadingManager:
    """
    One object to manage context for multi-threading.  This should make
    bin/swift less error-prone and allow us to test this code.
    """

    def __init__(self, create_connection, segment_threads=10, object_dd_threads=10, object_uu_threads=10, container_threads=10):
        """
        :param segment_threads: The number of threads allocated to segment
                                uploads
        :param object_dd_threads: The number of threads allocated to object
                                  download/delete jobs
        :param object_uu_threads: The number of threads allocated to object
                                  upload/update based jobs
        :param container_threads: The number of threads allocated to
                                  container/account level jobs
        """
        self.segment_pool = ConnectionThreadPoolExecutor(create_connection, max_workers=segment_threads)
        self.object_dd_pool = ConnectionThreadPoolExecutor(create_connection, max_workers=object_dd_threads)
        self.object_uu_pool = ConnectionThreadPoolExecutor(create_connection, max_workers=object_uu_threads)
        self.container_pool = ConnectionThreadPoolExecutor(create_connection, max_workers=container_threads)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.segment_pool.__exit__(exc_type, exc_value, traceback)
        self.object_dd_pool.__exit__(exc_type, exc_value, traceback)
        self.object_uu_pool.__exit__(exc_type, exc_value, traceback)
        self.container_pool.__exit__(exc_type, exc_value, traceback)