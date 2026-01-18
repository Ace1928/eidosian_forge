import ast
import base64
import itertools
import os
import pathlib
import signal
import struct
import tempfile
import threading
import time
import traceback
import json
import numpy as np
import pytest
import pyarrow as pa
from pyarrow.lib import IpcReadOptions, tobytes
from pyarrow.util import find_free_port
from pyarrow.tests import util
class HeadersTrailersFlightServer(FlightServerBase):

    def get_flight_info(self, context, descriptor):
        context.add_header('x-header', 'header-value')
        context.add_header('x-header-bin', 'header\x01value')
        context.add_trailer('x-trailer', 'trailer-value')
        context.add_trailer('x-trailer-bin', 'trailer\x01value')
        return flight.FlightInfo(pa.schema([]), descriptor, [], -1, -1)