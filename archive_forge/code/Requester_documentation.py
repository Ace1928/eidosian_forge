import io
import json
import logging
import mimetypes
import os
import re
import threading
import time
import urllib
import urllib.parse
from collections import deque
from datetime import datetime, timezone
from io import IOBase
from typing import (
import requests
import requests.adapters
from urllib3 import Retry
import github.Consts as Consts
import github.GithubException as GithubException

        Create a mutation in the format:
            mutation MutationName($input: MutationNameInput!) {
                mutationName(input: $input) {
                    <output>
                }
            }
        and call the self.graphql_query method
        