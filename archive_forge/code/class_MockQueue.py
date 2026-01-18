from __future__ import (absolute_import, division, print_function)
import asyncio
import json
import logging
from typing import Any, Dict
import re
class MockQueue:

    async def put(self, event):
        pass