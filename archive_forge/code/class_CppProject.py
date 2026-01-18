from __future__ import annotations
from mesonbuild.templates.sampleimpl import FileHeaderImpl
class CppProject(FileHeaderImpl):
    source_ext = 'cpp'
    header_ext = 'hpp'
    exe_template = hello_cpp_template
    exe_meson_template = hello_cpp_meson_template
    lib_template = lib_cpp_template
    lib_header_template = lib_hpp_template
    lib_test_template = lib_cpp_test_template
    lib_meson_template = lib_cpp_meson_template