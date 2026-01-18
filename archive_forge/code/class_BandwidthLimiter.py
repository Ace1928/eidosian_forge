import time
import threading
class BandwidthLimiter(object):

    def __init__(self, leaky_bucket, time_utils=None):
        """Limits bandwidth for shared S3 transfers

        :type leaky_bucket: LeakyBucket
        :param leaky_bucket: The leaky bucket to use limit bandwidth

        :type time_utils: TimeUtils
        :param time_utils: Time utility to use for interacting with time.
        """
        self._leaky_bucket = leaky_bucket
        self._time_utils = time_utils
        if time_utils is None:
            self._time_utils = TimeUtils()

    def get_bandwith_limited_stream(self, fileobj, transfer_coordinator, enabled=True):
        """Wraps a fileobj in a bandwidth limited stream wrapper

        :type fileobj: file-like obj
        :param fileobj: The file-like obj to wrap

        :type transfer_coordinator: s3transfer.futures.TransferCoordinator
        param transfer_coordinator: The coordinator for the general transfer
            that the wrapped stream is a part of

        :type enabled: boolean
        :param enabled: Whether bandwidth limiting should be enabled to start
        """
        stream = BandwidthLimitedStream(fileobj, self._leaky_bucket, transfer_coordinator, self._time_utils)
        if not enabled:
            stream.disable_bandwidth_limiting()
        return stream