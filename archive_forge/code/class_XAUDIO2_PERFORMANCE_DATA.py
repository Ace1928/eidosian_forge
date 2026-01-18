import ctypes
import platform
import os
from pyglet.libs.win32.constants import *
from pyglet.libs.win32.types import *
from pyglet.libs.win32 import com
from pyglet.util import debug_print
class XAUDIO2_PERFORMANCE_DATA(ctypes.Structure):
    _fields_ = [('AudioCyclesSinceLastQuery', c_uint64), ('TotalCyclesSinceLastQuery', c_uint64), ('MinimumCyclesPerQuantum', UINT32), ('MaximumCyclesPerQuantum', UINT32), ('MemoryUsageInBytes', UINT32), ('CurrentLatencyInSamples', UINT32), ('GlitchesSinceEngineStarted', UINT32), ('ActiveSourceVoiceCount', UINT32), ('TotalSourceVoiceCount', UINT32), ('ActiveSubmixVoiceCount', UINT32), ('ActiveResamplerCount', UINT32), ('ActiveMatrixMixCount', UINT32), ('ActiveXmaSourceVoices', UINT32), ('ActiveXmaStreams', UINT32)]

    def __repr__(self):
        return 'XAUDIO2PerformanceData(active_voices={}, total_voices={}, glitches={}, latency={} samples, memory_usage={} bytes)'.format(self.ActiveSourceVoiceCount, self.TotalSourceVoiceCount, self.GlitchesSinceEngineStarted, self.CurrentLatencyInSamples, self.MemoryUsageInBytes)