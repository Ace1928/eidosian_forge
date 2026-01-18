import ctypes
import platform
import os
from pyglet.libs.win32.constants import *
from pyglet.libs.win32.types import *
from pyglet.libs.win32 import com
from pyglet.util import debug_print
class IXAudio2(com.pIUnknown):
    _methods_ = [('RegisterForCallbacks', com.STDMETHOD(POINTER(IXAudio2EngineCallback))), ('UnregisterForCallbacks', com.VOIDMETHOD(POINTER(IXAudio2EngineCallback))), ('CreateSourceVoice', com.STDMETHOD(POINTER(IXAudio2SourceVoice), POINTER(WAVEFORMATEX), UINT32, c_float, POINTER(IXAudio2VoiceCallback), POINTER(XAUDIO2_VOICE_SENDS), POINTER(XAUDIO2_EFFECT_CHAIN))), ('CreateSubmixVoice', com.STDMETHOD(POINTER(IXAudio2SubmixVoice), UINT32, UINT32, UINT32, UINT32, POINTER(XAUDIO2_VOICE_SENDS), POINTER(XAUDIO2_EFFECT_CHAIN))), ('CreateMasteringVoice', com.STDMETHOD(POINTER(IXAudio2MasteringVoice), UINT32, UINT32, UINT32, LPCWSTR, POINTER(XAUDIO2_EFFECT_CHAIN), UINT32)), ('StartEngine', com.STDMETHOD()), ('StopEngine', com.VOIDMETHOD()), ('CommitChanges', com.STDMETHOD(UINT32)), ('GetPerformanceData', com.VOIDMETHOD(POINTER(XAUDIO2_PERFORMANCE_DATA))), ('SetDebugConfiguration', com.VOIDMETHOD(POINTER(XAUDIO2_DEBUG_CONFIGURATION), c_void_p))]