import os
import re
import sys
import pretty_vcproj
def PrintVCProj(projects):
    for project in projects:
        print('-------------------------------------')
        print('-------------------------------------')
        print(project)
        print(project)
        print(project)
        print('-------------------------------------')
        print('-------------------------------------')
        project_path = os.path.abspath(os.path.join(os.path.dirname(sys.argv[1]), projects[project][2]))
        pretty = pretty_vcproj
        argv = ['', project_path, '$(SolutionDir)=%s\\' % os.path.dirname(sys.argv[1])]
        argv.extend(sys.argv[3:])
        pretty.main(argv)