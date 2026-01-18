from __future__ import annotations
from mesonbuild.templates.sampleimpl import FileHeaderImpl
class CProject(FileHeaderImpl):
    source_ext = 'c'
    header_ext = 'h'
    exe_template = hello_c_template
    exe_meson_template = hello_c_meson_template
    lib_template = lib_c_template
    lib_header_template = lib_h_template
    lib_test_template = lib_c_test_template
    lib_meson_template = lib_c_meson_template