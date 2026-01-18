import dataclasses
import json
import logging
import socket
import sys
import threading
import traceback
import urllib.parse
from collections import defaultdict, deque
from copy import deepcopy
from typing import (
import flask
import pandas as pd
import requests
import responses
import wandb
import wandb.util
from wandb.sdk.lib.timer import Timer
class RelayControlProtocol(Protocol):

    def process(self, request: 'flask.Request') -> None:
        ...

    def control(self, request: 'flask.Request') -> Mapping[str, str]:
        ...