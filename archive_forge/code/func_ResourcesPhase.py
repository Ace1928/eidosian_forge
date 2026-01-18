import gyp.common
from functools import cmp_to_key
import hashlib
from operator import attrgetter
import posixpath
import re
import struct
import sys
def ResourcesPhase(self):
    resources_phase = self.GetBuildPhaseByType(PBXResourcesBuildPhase)
    if resources_phase is None:
        resources_phase = PBXResourcesBuildPhase()
        insert_at = len(self._properties['buildPhases'])
        for index, phase in enumerate(self._properties['buildPhases']):
            if isinstance(phase, PBXSourcesBuildPhase) or isinstance(phase, PBXFrameworksBuildPhase):
                insert_at = index
                break
        self._properties['buildPhases'].insert(insert_at, resources_phase)
        resources_phase.parent = self
    return resources_phase