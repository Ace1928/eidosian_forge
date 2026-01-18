import os
import shlex
import sys
from pbr import find_package
from pbr.hooks import base
class FilesConfig(base.BaseConfig):
    section = 'files'

    def __init__(self, config, name):
        super(FilesConfig, self).__init__(config)
        self.name = name
        self.data_files = self.config.get('data_files', '')

    def save(self):
        self.config['data_files'] = self.data_files
        super(FilesConfig, self).save()

    def expand_globs(self):
        finished = []
        for line in self.data_files.split('\n'):
            if line.rstrip().endswith('*') and '=' in line:
                target, source_glob = line.split('=')
                source_prefix = source_glob.strip()[:-1]
                target = target.strip()
                if not target.endswith(os.path.sep):
                    target += os.path.sep
                unquoted_prefix = unquote_path(source_prefix)
                unquoted_target = unquote_path(target)
                for dirpath, dirnames, fnames in os.walk(unquoted_prefix):
                    new_prefix = dirpath.replace(unquoted_prefix, unquoted_target, 1)
                    finished.append("'%s' = " % new_prefix)
                    finished.extend([" '%s'" % os.path.join(dirpath, f) for f in fnames])
            else:
                finished.append(line)
        self.data_files = '\n'.join(finished)

    def add_man_path(self, man_path):
        self.data_files = "%s\n'%s' =" % (self.data_files, man_path)

    def add_man_page(self, man_page):
        self.data_files = "%s\n  '%s'" % (self.data_files, man_page)

    def get_man_sections(self):
        man_sections = dict()
        manpages = self.pbr_config['manpages']
        for manpage in manpages.split():
            section_number = manpage.strip()[-1]
            section = man_sections.get(section_number, list())
            section.append(manpage.strip())
            man_sections[section_number] = section
        return man_sections

    def hook(self):
        packages = self.config.get('packages', self.name).strip()
        expanded = []
        for pkg in packages.split('\n'):
            if os.path.isdir(pkg.strip()):
                expanded.append(find_package.smart_find_packages(pkg.strip()))
        self.config['packages'] = '\n'.join(expanded)
        self.expand_globs()
        if 'manpages' in self.pbr_config:
            man_sections = self.get_man_sections()
            for section, pages in man_sections.items():
                manpath = get_man_section(section)
                self.add_man_path(manpath)
                for page in pages:
                    self.add_man_page(page)