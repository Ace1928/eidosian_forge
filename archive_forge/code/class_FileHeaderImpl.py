from __future__ import annotations
import abc
import re
import typing as T
class FileHeaderImpl(FileImpl):

    @abc.abstractproperty
    def header_ext(self) -> str:
        pass

    @abc.abstractproperty
    def lib_header_template(self) -> str:
        pass

    def lib_kwargs(self) -> T.Dict[str, str]:
        kwargs = super().lib_kwargs()
        kwargs['header_file'] = f'{self.lowercase_token}.{self.header_ext}'
        return kwargs

    def create_library(self) -> None:
        super().create_library()
        kwargs = self.lib_kwargs()
        with open(kwargs['header_file'], 'w', encoding='utf-8') as f:
            f.write(self.lib_header_template.format_map(kwargs))