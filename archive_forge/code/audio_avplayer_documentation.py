from kivy.core.audio import Sound, SoundLoader
from pyobjus import autoclass
from pyobjus.dylib_manager import load_framework, INCLUDE

AudioAvplayer: implementation of Sound using pyobjus / AVFoundation.
Works on iOS / OSX.
