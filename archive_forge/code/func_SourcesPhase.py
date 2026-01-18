import gyp.common
from functools import cmp_to_key
import hashlib
from operator import attrgetter
import posixpath
import re
import struct
import sys
def SourcesPhase(self):
    sources_phase = self.GetBuildPhaseByType(PBXSourcesBuildPhase)
    if sources_phase is None:
        sources_phase = PBXSourcesBuildPhase()
        self.AppendProperty('buildPhases', sources_phase)
    return sources_phase