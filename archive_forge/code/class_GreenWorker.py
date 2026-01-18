import sys
from futurist import _utils
class GreenWorker(object):

    def __init__(self, work, work_queue):
        self.work = work
        self.work_queue = work_queue

    def __call__(self):
        try:
            self.work.run()
        except SystemExit as e:
            exc_info = sys.exc_info()
            try:
                while True:
                    try:
                        w = self.work_queue.get_nowait()
                    except greenqueue.Empty:
                        break
                    try:
                        w.fail(exc_info)
                    finally:
                        self.work_queue.task_done()
            finally:
                del exc_info
                raise e
        while True:
            try:
                w = self.work_queue.get_nowait()
            except greenqueue.Empty:
                break
            else:
                try:
                    w.run()
                finally:
                    self.work_queue.task_done()