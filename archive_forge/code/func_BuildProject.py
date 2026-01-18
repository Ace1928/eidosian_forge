import os
import re
import sys
import pretty_vcproj
def BuildProject(project, built, projects, deps):
    for dep in deps[project]:
        if dep not in built:
            BuildProject(dep, built, projects, deps)
    print(project)
    built.append(project)