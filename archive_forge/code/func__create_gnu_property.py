from ..construct import (
from ..common.construct_utils import ULEB128
from ..common.utils import roundup
from .enums import *
def _create_gnu_property(self):

    def roundup_padding(ctx):
        if self.elfclass == 32:
            return roundup(ctx.pr_datasz, 2) - ctx.pr_datasz
        return roundup(ctx.pr_datasz, 3) - ctx.pr_datasz

    def classify_pr_data(ctx):
        if type(ctx.pr_type) is not str:
            return None
        if ctx.pr_type.startswith('GNU_PROPERTY_X86_'):
            return ('GNU_PROPERTY_X86_*', 4, 0)
        elif ctx.pr_type.startswith('GNU_PROPERTY_AARCH64_'):
            return ('GNU_PROPERTY_AARCH64_*', 4, 0)
        return (ctx.pr_type, ctx.pr_datasz, self.elfclass)
    self.Elf_Prop = Struct('Elf_Prop', Enum(self.Elf_word('pr_type'), **ENUM_NOTE_GNU_PROPERTY_TYPE), self.Elf_word('pr_datasz'), Switch('pr_data', classify_pr_data, {('GNU_PROPERTY_STACK_SIZE', 4, 32): self.Elf_word('pr_data'), ('GNU_PROPERTY_STACK_SIZE', 8, 64): self.Elf_word64('pr_data'), ('GNU_PROPERTY_X86_*', 4, 0): self.Elf_word('pr_data'), ('GNU_PROPERTY_AARCH64_*', 4, 0): self.Elf_word('pr_data')}, default=Field('pr_data', lambda ctx: ctx.pr_datasz)), Padding(roundup_padding))