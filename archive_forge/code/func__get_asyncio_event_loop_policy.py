import asyncio
import sys
from asyncio import AbstractEventLoop, AbstractEventLoopPolicy
from contextlib import suppress
from typing import Any, Callable, Dict, Optional, Sequence, Type
from warnings import catch_warnings, filterwarnings, warn
from twisted.internet import asyncioreactor, error
from twisted.internet.base import DelayedCall
from scrapy.exceptions import ScrapyDeprecationWarning
from scrapy.utils.misc import load_object
def _get_asyncio_event_loop_policy() -> AbstractEventLoopPolicy:
    policy = asyncio.get_event_loop_policy()
    if sys.version_info >= (3, 8) and sys.platform == 'win32' and (not isinstance(policy, asyncio.WindowsSelectorEventLoopPolicy)):
        policy = asyncio.WindowsSelectorEventLoopPolicy()
        asyncio.set_event_loop_policy(policy)
    return policy