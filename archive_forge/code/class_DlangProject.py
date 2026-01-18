from __future__ import annotations
from mesonbuild.templates.sampleimpl import FileImpl
import typing as T
import std.stdio;
import std.stdio;
import {module_file};
class DlangProject(FileImpl):
    source_ext = 'd'
    exe_template = hello_d_template
    exe_meson_template = hello_d_meson_template
    lib_template = lib_d_template
    lib_test_template = lib_d_test_template
    lib_meson_template = lib_d_meson_template

    def lib_kwargs(self) -> T.Dict[str, str]:
        kwargs = super().lib_kwargs()
        kwargs['module_file'] = self.lowercase_token
        return kwargs