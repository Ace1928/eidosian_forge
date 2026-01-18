import io
import functools
import logging
import time
import warnings
import smart_open.bytebuffer
import smart_open.concurrency
import smart_open.utils
from smart_open import constants
def _upload_next_part(self):
    part_num = self._total_parts + 1
    logger.info('%s: uploading part_num: %i, %i bytes (total %.3fGB)', self, part_num, self._buf.tell(), self._total_bytes / 1024.0 ** 3)
    self._buf.seek(0)
    upload = _retry_if_failed(functools.partial(self._client.upload_part, Bucket=self._bucket, Key=self._key, UploadId=self._upload_id, PartNumber=part_num, Body=self._buf))
    self._parts.append({'ETag': upload['ETag'], 'PartNumber': part_num})
    logger.debug('%s: upload of part_num #%i finished', self, part_num)
    self._total_parts += 1
    self._buf.seek(0)
    self._buf.truncate(0)