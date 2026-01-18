from __future__ import annotations
import asyncio
import logging
import os
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Union
import aiohttp
import requests
from aiohttp import ServerTimeoutError
from langchain_core.pydantic_v1 import BaseModel, Field, root_validator, validator
from requests.exceptions import Timeout
def _get_tables_todo(self, tables_todo: List[str]) -> List[str]:
    """Get the tables that still need to be queried."""
    return [table for table in tables_todo if table not in self.schemas]