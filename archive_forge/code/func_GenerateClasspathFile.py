from xml.sax.saxutils import escape
import os.path
import subprocess
import gyp
import gyp.common
import gyp.msvs_emulation
import shlex
import xml.etree.cElementTree as ET
def GenerateClasspathFile(target_list, target_dicts, toplevel_dir, toplevel_build, out_name):
    """Generates a classpath file suitable for symbol navigation and code
  completion of Java code (such as in Android projects) by finding all
  .java and .jar files used as action inputs."""
    gyp.common.EnsureDirExists(out_name)
    result = ET.Element('classpath')

    def AddElements(kind, paths):
        rel_paths = set()
        for path in paths:
            if os.path.isabs(path):
                rel_paths.add(os.path.relpath(path, toplevel_dir))
            else:
                rel_paths.add(path)
        for path in sorted(rel_paths):
            entry_element = ET.SubElement(result, 'classpathentry')
            entry_element.set('kind', kind)
            entry_element.set('path', path)
    AddElements('lib', GetJavaJars(target_list, target_dicts, toplevel_dir))
    AddElements('src', GetJavaSourceDirs(target_list, target_dicts, toplevel_dir))
    AddElements('con', ['org.eclipse.jdt.launching.JRE_CONTAINER'])
    AddElements('output', [os.path.join(toplevel_build, '.eclipse-java-build')])
    ET.ElementTree(result).write(out_name)