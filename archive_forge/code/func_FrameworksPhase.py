import gyp.common
from functools import cmp_to_key
import hashlib
from operator import attrgetter
import posixpath
import re
import struct
import sys
def FrameworksPhase(self):
    frameworks_phase = self.GetBuildPhaseByType(PBXFrameworksBuildPhase)
    if frameworks_phase is None:
        frameworks_phase = PBXFrameworksBuildPhase()
        self.AppendProperty('buildPhases', frameworks_phase)
    return frameworks_phase