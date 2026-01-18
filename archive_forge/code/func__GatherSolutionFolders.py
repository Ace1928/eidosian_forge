import ntpath
import os
import posixpath
import re
import subprocess
import sys
from collections import OrderedDict
import gyp.common
import gyp.easy_xml as easy_xml
import gyp.generator.ninja as ninja_generator
import gyp.MSVSNew as MSVSNew
import gyp.MSVSProject as MSVSProject
import gyp.MSVSSettings as MSVSSettings
import gyp.MSVSToolFile as MSVSToolFile
import gyp.MSVSUserFile as MSVSUserFile
import gyp.MSVSUtil as MSVSUtil
import gyp.MSVSVersion as MSVSVersion
from gyp.common import GypError
from gyp.common import OrderedSet
def _GatherSolutionFolders(sln_projects, project_objects, flat):
    root = {}
    for p in sln_projects:
        gyp_file, target = gyp.common.ParseQualifiedTarget(p)[0:2]
        if p.endswith('#host'):
            target += '_host'
        gyp_dir = os.path.dirname(gyp_file)
        path_dict = _GetPathDict(root, gyp_dir)
        path_dict[target + '.vcproj'] = project_objects[p]
    while len(root) == 1 and type(root[next(iter(root))]) == dict:
        root = root[next(iter(root))]
    root = _CollapseSingles('', root)
    return _DictsToFolders('', root, flat)