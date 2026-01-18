import base64
import functools
import itertools
import logging
import os
import queue
import random
import sys
import threading
import time
from types import TracebackType
from typing import (
import requests
import wandb
from wandb import util
from wandb.sdk.internal import internal_api
from ..lib import file_stream_utils
class JsonlFilePolicy(DefaultFilePolicy):

    def process_chunks(self, chunks: List[Chunk]) -> 'ProcessedChunk':
        chunk_id = self._chunk_id
        self._chunk_id += len(chunks)
        chunk_data = []
        for chunk in chunks:
            if len(chunk.data) > util.MAX_LINE_BYTES:
                msg = 'Metric data exceeds maximum size of {} ({})'.format(util.to_human_size(util.MAX_LINE_BYTES), util.to_human_size(len(chunk.data)))
                wandb.termerror(msg, repeat=False)
                wandb._sentry.message(msg, repeat=False)
            else:
                chunk_data.append(chunk.data)
        return {'offset': chunk_id, 'content': chunk_data}