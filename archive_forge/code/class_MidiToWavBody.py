import io
import global_var
from fastapi import APIRouter, HTTPException, UploadFile, status
from starlette.responses import StreamingResponse
from pydantic import BaseModel
from utils.midi import *
from midi2audio import FluidSynth
class MidiToWavBody(BaseModel):
    midi_path: str
    wav_path: str
    sound_font_path: str = 'assets/default_sound_font.sf2'
    model_config = {'json_schema_extra': {'example': {'midi_path': 'midi/sample.mid', 'wav_path': 'midi/sample.wav', 'sound_font_path': 'assets/default_sound_font.sf2'}}}