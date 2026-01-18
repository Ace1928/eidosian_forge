import socket
import time
class CallbackTestHarness(object):

    def __init__(self, fail_after_n_bytes=0, num_times_to_fail=1, exception=socket.error('mock socket error', 0), fp_to_change=None, fp_change_pos=None, delay_after_change=None):
        self.fail_after_n_bytes = fail_after_n_bytes
        self.num_times_to_fail = num_times_to_fail
        self.exception = exception
        self.fp_to_change = fp_to_change
        self.fp_change_pos = fp_change_pos
        self.delay_after_change = delay_after_change
        self.num_failures = 0
        self.transferred_seq_before_first_failure = []
        self.transferred_seq_after_first_failure = []

    def call(self, total_bytes_transferred, unused_total_size):
        """
        To use this test harness, pass the 'call' method of the instantiated
        object as the cb param to the set_contents_from_file() or
        get_contents_to_file() call.
        """
        if self.num_failures:
            self.transferred_seq_after_first_failure.append(total_bytes_transferred)
        else:
            self.transferred_seq_before_first_failure.append(total_bytes_transferred)
        if total_bytes_transferred >= self.fail_after_n_bytes and self.num_failures < self.num_times_to_fail:
            self.num_failures += 1
            if self.fp_to_change and self.fp_change_pos is not None:
                cur_pos = self.fp_to_change.tell()
                self.fp_to_change.seek(self.fp_change_pos)
                self.fp_to_change.write('abc')
                self.fp_to_change.seek(cur_pos)
                if self.delay_after_change:
                    time.sleep(self.delay_after_change)
            self.called = True
            raise self.exception