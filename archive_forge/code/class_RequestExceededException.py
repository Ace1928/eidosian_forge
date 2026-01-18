import time
import threading
class RequestExceededException(Exception):

    def __init__(self, requested_amt, retry_time):
        """Error when requested amount exceeds what is allowed

        The request that raised this error should be retried after waiting
        the time specified by ``retry_time``.

        :type requested_amt: int
        :param requested_amt: The originally requested byte amount

        :type retry_time: float
        :param retry_time: The length in time to wait to retry for the
            requested amount
        """
        self.requested_amt = requested_amt
        self.retry_time = retry_time
        msg = 'Request amount %s exceeded the amount available. Retry in %s' % (requested_amt, retry_time)
        super(RequestExceededException, self).__init__(msg)