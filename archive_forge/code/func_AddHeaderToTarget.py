import filecmp
import gyp.common
import gyp.xcodeproj_file
import gyp.xcode_ninja
import errno
import os
import sys
import posixpath
import re
import shutil
import subprocess
import tempfile
def AddHeaderToTarget(header, pbxp, xct, is_public):
    settings = '{ATTRIBUTES = (%s, ); }' % ('Private', 'Public')[is_public]
    xct.HeadersPhase().AddFile(header, settings)