from typing import Any, Dict, List, Union
from utils.log import quick_log
from fastapi import APIRouter, HTTPException, Request, Response, status
from pydantic import BaseModel
import gc
import copy
import global_var
class LongestPrefixStateBody(BaseModel):
    prompt: str