from xml.sax.saxutils import escape
import os.path
import subprocess
import gyp
import gyp.common
import gyp.msvs_emulation
import shlex
import xml.etree.cElementTree as ET
def WriteIncludePaths(out, eclipse_langs, include_dirs):
    """Write the includes section of a CDT settings export file."""
    out.write('  <section name="org.eclipse.cdt.internal.ui.wizards.settingswizards.IncludePaths">\n')
    out.write('    <language name="holder for library settings"></language>\n')
    for lang in eclipse_langs:
        out.write('    <language name="%s">\n' % lang)
        for include_dir in include_dirs:
            out.write('      <includepath workspace_path="false">%s</includepath>\n' % include_dir)
        out.write('    </language>\n')
    out.write('  </section>\n')