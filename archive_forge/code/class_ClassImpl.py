from __future__ import annotations
import abc
import re
import typing as T
class ClassImpl(SampleImpl):
    """For Class based languages, like Java and C#"""

    def create_executable(self) -> None:
        source_name = f'{self.capitalized_token}.{self.source_ext}'
        with open(source_name, 'w', encoding='utf-8') as f:
            f.write(self.exe_template.format(project_name=self.name, class_name=self.capitalized_token))
        with open('meson.build', 'w', encoding='utf-8') as f:
            f.write(self.exe_meson_template.format(project_name=self.name, exe_name=self.name, source_name=source_name, version=self.version))

    def create_library(self) -> None:
        lib_name = f'{self.capitalized_token}.{self.source_ext}'
        test_name = f'{self.capitalized_token}_test.{self.source_ext}'
        kwargs = {'utoken': self.uppercase_token, 'ltoken': self.lowercase_token, 'class_test': f'{self.capitalized_token}_test', 'class_name': self.capitalized_token, 'source_file': lib_name, 'test_source_file': test_name, 'test_exe_name': f'{self.lowercase_token}_test', 'project_name': self.name, 'lib_name': self.lowercase_token, 'test_name': self.lowercase_token, 'version': self.version}
        with open(lib_name, 'w', encoding='utf-8') as f:
            f.write(self.lib_template.format(**kwargs))
        with open(test_name, 'w', encoding='utf-8') as f:
            f.write(self.lib_test_template.format(**kwargs))
        with open('meson.build', 'w', encoding='utf-8') as f:
            f.write(self.lib_meson_template.format(**kwargs))