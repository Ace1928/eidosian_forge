import multiprocessing
import os
import signal
import subprocess
import gyp.common
import gyp.xcode_emulation
class CMakeTargetType:

    def __init__(self, command, modifier, property_modifier):
        self.command = command
        self.modifier = modifier
        self.property_modifier = property_modifier