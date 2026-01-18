import time
import threading
def _release_requested_amt_for_scheduled_request(self, amt, request_token, time_now):
    self._consumption_scheduler.process_scheduled_consumption(request_token)
    return self._release_requested_amt(amt, time_now)