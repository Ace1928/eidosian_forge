import pathlib
from utils.log import quick_log
from fastapi import APIRouter, HTTPException, Request, Response, status as Status
from pydantic import BaseModel
from utils.rwkv import *
from utils.torch import *
import global_var
class SwitchModelBody(BaseModel):
    model: str
    strategy: str
    tokenizer: Union[str, None] = None
    customCuda: bool = False
    deploy: bool = Field(False, description='Deploy mode. If success, will disable /switch-model, /exit and other dangerous APIs (state cache APIs, part of midi APIs)')
    model_config = {'json_schema_extra': {'example': {'model': 'models/RWKV-4-World-3B-v1-20230619-ctx4096.pth', 'strategy': 'cuda fp16', 'tokenizer': '', 'customCuda': False, 'deploy': False}}}