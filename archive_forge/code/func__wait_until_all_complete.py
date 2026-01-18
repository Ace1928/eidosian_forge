import copy
import logging
from s3transfer.utils import get_callbacks
def _wait_until_all_complete(self, futures):
    logger.debug('%s about to wait for the following futures %s', self, futures)
    for future in futures:
        try:
            logger.debug('%s about to wait for %s', self, future)
            future.result()
        except Exception:
            pass
    logger.debug('%s done waiting for dependent futures', self)