import io
import global_var
from fastapi import APIRouter, HTTPException, UploadFile, status
from starlette.responses import StreamingResponse
from pydantic import BaseModel
from utils.midi import *
from midi2audio import FluidSynth
class TxtToMidiBody(BaseModel):
    txt_path: str
    midi_path: str
    model_config = {'json_schema_extra': {'example': {'txt_path': 'midi/sample.txt', 'midi_path': 'midi/sample.mid'}}}