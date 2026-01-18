from collections import namedtuple
from ..common.exceptions import ELFRelocationError
from ..common.utils import elf_assert, struct_parse
from .sections import Section
from .enums import (
from ..construct import Container
class RelocationHandler(object):
    """ Handles the logic of relocations in ELF files.
    """

    def __init__(self, elffile):
        self.elffile = elffile

    def find_relocations_for_section(self, section):
        """ Given a section, find the relocation section for it in the ELF
            file. Return a RelocationSection object, or None if none was
            found.
        """
        reloc_section_names = ('.rel' + section.name, '.rela' + section.name)
        for relsection in self.elffile.iter_sections():
            if isinstance(relsection, RelocationSection) and relsection.name in reloc_section_names:
                return relsection
        return None

    def apply_section_relocations(self, stream, reloc_section):
        """ Apply all relocations in reloc_section (a RelocationSection object)
            to the given stream, that contains the data of the section that is
            being relocated. The stream is modified as a result.
        """
        symtab = self.elffile.get_section(reloc_section['sh_link'])
        for reloc in reloc_section.iter_relocations():
            self._do_apply_relocation(stream, reloc, symtab)

    def _do_apply_relocation(self, stream, reloc, symtab):
        if reloc['r_info_sym'] >= symtab.num_symbols():
            raise ELFRelocationError('Invalid symbol reference in relocation: index %s' % reloc['r_info_sym'])
        sym_value = symtab.get_symbol(reloc['r_info_sym'])['st_value']
        reloc_type = reloc['r_info_type']
        recipe = None
        if self.elffile.get_machine_arch() == 'x86':
            if reloc.is_RELA():
                raise ELFRelocationError('Unexpected RELA relocation for x86: %s' % reloc)
            recipe = self._RELOCATION_RECIPES_X86.get(reloc_type, None)
        elif self.elffile.get_machine_arch() == 'x64':
            if not reloc.is_RELA():
                raise ELFRelocationError('Unexpected REL relocation for x64: %s' % reloc)
            recipe = self._RELOCATION_RECIPES_X64.get(reloc_type, None)
        elif self.elffile.get_machine_arch() == 'MIPS':
            if reloc.is_RELA():
                if reloc_type == ENUM_RELOC_TYPE_MIPS['R_MIPS_64']:
                    if reloc['r_type2'] != 0 or reloc['r_type3'] != 0 or reloc['r_ssym'] != 0:
                        raise ELFRelocationError('Multiple relocations in R_MIPS_64 are not implemented: %s' % reloc)
                recipe = self._RELOCATION_RECIPES_MIPS_RELA.get(reloc_type, None)
            else:
                recipe = self._RELOCATION_RECIPES_MIPS_REL.get(reloc_type, None)
        elif self.elffile.get_machine_arch() == 'ARM':
            if reloc.is_RELA():
                raise ELFRelocationError('Unexpected RELA relocation for ARM: %s' % reloc)
            recipe = self._RELOCATION_RECIPES_ARM.get(reloc_type, None)
        elif self.elffile.get_machine_arch() == 'AArch64':
            recipe = self._RELOCATION_RECIPES_AARCH64.get(reloc_type, None)
        elif self.elffile.get_machine_arch() == '64-bit PowerPC':
            recipe = self._RELOCATION_RECIPES_PPC64.get(reloc_type, None)
        elif self.elffile.get_machine_arch() == 'IBM S/390':
            recipe = self._RELOCATION_RECIPES_S390X.get(reloc_type, None)
        elif self.elffile.get_machine_arch() == 'Linux BPF - in-kernel virtual machine':
            recipe = self._RELOCATION_RECIPES_EBPF.get(reloc_type, None)
        elif self.elffile.get_machine_arch() == 'LoongArch':
            if not reloc.is_RELA():
                raise ELFRelocationError('Unexpected REL relocation for LoongArch: %s' % reloc)
            recipe = self._RELOCATION_RECIPES_LOONGARCH.get(reloc_type, None)
        if recipe is None:
            raise ELFRelocationError('Unsupported relocation type: %s' % reloc_type)
        if recipe.bytesize == 4:
            value_struct = self.elffile.structs.Elf_word('')
        elif recipe.bytesize == 8:
            value_struct = self.elffile.structs.Elf_word64('')
        elif recipe.bytesize == 1:
            value_struct = self.elffile.structs.Elf_byte('')
        elif recipe.bytesize == 2:
            value_struct = self.elffile.structs.Elf_half('')
        else:
            raise ELFRelocationError('Invalid bytesize %s for relocation' % recipe.bytesize)
        original_value = struct_parse(value_struct, stream, stream_pos=reloc['r_offset'])
        relocated_value = recipe.calc_func(value=original_value, sym_value=sym_value, offset=reloc['r_offset'], addend=reloc['r_addend'] if recipe.has_addend else 0)
        stream.seek(reloc['r_offset'])
        relocated_value = relocated_value % 2 ** (recipe.bytesize * 8)
        value_struct.build_stream(relocated_value, stream)
    _RELOCATION_RECIPE_TYPE = namedtuple('_RELOCATION_RECIPE_TYPE', 'bytesize has_addend calc_func')

    def _reloc_calc_identity(value, sym_value, offset, addend=0):
        return value

    def _reloc_calc_sym_plus_value(value, sym_value, offset, addend=0):
        return sym_value + value + addend

    def _reloc_calc_sym_plus_value_pcrel(value, sym_value, offset, addend=0):
        return sym_value + value - offset

    def _reloc_calc_sym_plus_addend(value, sym_value, offset, addend=0):
        return sym_value + addend

    def _reloc_calc_sym_plus_addend_pcrel(value, sym_value, offset, addend=0):
        return sym_value + addend - offset

    def _reloc_calc_value_minus_sym_addend(value, sym_value, offset, addend=0):
        return value - sym_value - addend

    def _arm_reloc_calc_sym_plus_value_pcrel(value, sym_value, offset, addend=0):
        return sym_value // 4 + value - offset // 4

    def _bpf_64_32_reloc_calc_sym_plus_addend(value, sym_value, offset, addend=0):
        return (sym_value + addend) // 8 - 1
    _RELOCATION_RECIPES_ARM = {ENUM_RELOC_TYPE_ARM['R_ARM_ABS32']: _RELOCATION_RECIPE_TYPE(bytesize=4, has_addend=False, calc_func=_reloc_calc_sym_plus_value), ENUM_RELOC_TYPE_ARM['R_ARM_CALL']: _RELOCATION_RECIPE_TYPE(bytesize=4, has_addend=False, calc_func=_arm_reloc_calc_sym_plus_value_pcrel)}
    _RELOCATION_RECIPES_AARCH64 = {ENUM_RELOC_TYPE_AARCH64['R_AARCH64_ABS64']: _RELOCATION_RECIPE_TYPE(bytesize=8, has_addend=True, calc_func=_reloc_calc_sym_plus_addend), ENUM_RELOC_TYPE_AARCH64['R_AARCH64_ABS32']: _RELOCATION_RECIPE_TYPE(bytesize=4, has_addend=True, calc_func=_reloc_calc_sym_plus_addend), ENUM_RELOC_TYPE_AARCH64['R_AARCH64_PREL32']: _RELOCATION_RECIPE_TYPE(bytesize=4, has_addend=True, calc_func=_reloc_calc_sym_plus_addend_pcrel)}
    _RELOCATION_RECIPES_MIPS_REL = {ENUM_RELOC_TYPE_MIPS['R_MIPS_NONE']: _RELOCATION_RECIPE_TYPE(bytesize=4, has_addend=False, calc_func=_reloc_calc_identity), ENUM_RELOC_TYPE_MIPS['R_MIPS_32']: _RELOCATION_RECIPE_TYPE(bytesize=4, has_addend=False, calc_func=_reloc_calc_sym_plus_value)}
    _RELOCATION_RECIPES_MIPS_RELA = {ENUM_RELOC_TYPE_MIPS['R_MIPS_NONE']: _RELOCATION_RECIPE_TYPE(bytesize=4, has_addend=True, calc_func=_reloc_calc_identity), ENUM_RELOC_TYPE_MIPS['R_MIPS_32']: _RELOCATION_RECIPE_TYPE(bytesize=4, has_addend=True, calc_func=_reloc_calc_sym_plus_value), ENUM_RELOC_TYPE_MIPS['R_MIPS_64']: _RELOCATION_RECIPE_TYPE(bytesize=8, has_addend=True, calc_func=_reloc_calc_sym_plus_value)}
    _RELOCATION_RECIPES_PPC64 = {ENUM_RELOC_TYPE_PPC64['R_PPC64_ADDR32']: _RELOCATION_RECIPE_TYPE(bytesize=4, has_addend=True, calc_func=_reloc_calc_sym_plus_addend), ENUM_RELOC_TYPE_PPC64['R_PPC64_REL32']: _RELOCATION_RECIPE_TYPE(bytesize=4, has_addend=True, calc_func=_reloc_calc_sym_plus_addend_pcrel), ENUM_RELOC_TYPE_PPC64['R_PPC64_ADDR64']: _RELOCATION_RECIPE_TYPE(bytesize=8, has_addend=True, calc_func=_reloc_calc_sym_plus_addend)}
    _RELOCATION_RECIPES_X86 = {ENUM_RELOC_TYPE_i386['R_386_NONE']: _RELOCATION_RECIPE_TYPE(bytesize=4, has_addend=False, calc_func=_reloc_calc_identity), ENUM_RELOC_TYPE_i386['R_386_32']: _RELOCATION_RECIPE_TYPE(bytesize=4, has_addend=False, calc_func=_reloc_calc_sym_plus_value), ENUM_RELOC_TYPE_i386['R_386_PC32']: _RELOCATION_RECIPE_TYPE(bytesize=4, has_addend=False, calc_func=_reloc_calc_sym_plus_value_pcrel)}
    _RELOCATION_RECIPES_X64 = {ENUM_RELOC_TYPE_x64['R_X86_64_NONE']: _RELOCATION_RECIPE_TYPE(bytesize=8, has_addend=True, calc_func=_reloc_calc_identity), ENUM_RELOC_TYPE_x64['R_X86_64_64']: _RELOCATION_RECIPE_TYPE(bytesize=8, has_addend=True, calc_func=_reloc_calc_sym_plus_addend), ENUM_RELOC_TYPE_x64['R_X86_64_PC32']: _RELOCATION_RECIPE_TYPE(bytesize=4, has_addend=True, calc_func=_reloc_calc_sym_plus_addend_pcrel), ENUM_RELOC_TYPE_x64['R_X86_64_32']: _RELOCATION_RECIPE_TYPE(bytesize=4, has_addend=True, calc_func=_reloc_calc_sym_plus_addend), ENUM_RELOC_TYPE_x64['R_X86_64_32S']: _RELOCATION_RECIPE_TYPE(bytesize=4, has_addend=True, calc_func=_reloc_calc_sym_plus_addend)}
    _RELOCATION_RECIPES_EBPF = {ENUM_RELOC_TYPE_BPF['R_BPF_NONE']: _RELOCATION_RECIPE_TYPE(bytesize=8, has_addend=False, calc_func=_reloc_calc_identity), ENUM_RELOC_TYPE_BPF['R_BPF_64_64']: _RELOCATION_RECIPE_TYPE(bytesize=8, has_addend=False, calc_func=_reloc_calc_identity), ENUM_RELOC_TYPE_BPF['R_BPF_64_32']: _RELOCATION_RECIPE_TYPE(bytesize=8, has_addend=False, calc_func=_bpf_64_32_reloc_calc_sym_plus_addend), ENUM_RELOC_TYPE_BPF['R_BPF_64_NODYLD32']: _RELOCATION_RECIPE_TYPE(bytesize=4, has_addend=False, calc_func=_reloc_calc_identity), ENUM_RELOC_TYPE_BPF['R_BPF_64_ABS64']: _RELOCATION_RECIPE_TYPE(bytesize=8, has_addend=False, calc_func=_reloc_calc_identity), ENUM_RELOC_TYPE_BPF['R_BPF_64_ABS32']: _RELOCATION_RECIPE_TYPE(bytesize=4, has_addend=False, calc_func=_reloc_calc_identity)}
    _RELOCATION_RECIPES_LOONGARCH = {ENUM_RELOC_TYPE_LOONGARCH['R_LARCH_NONE']: _RELOCATION_RECIPE_TYPE(bytesize=4, has_addend=False, calc_func=_reloc_calc_identity), ENUM_RELOC_TYPE_LOONGARCH['R_LARCH_32']: _RELOCATION_RECIPE_TYPE(bytesize=4, has_addend=True, calc_func=_reloc_calc_sym_plus_addend), ENUM_RELOC_TYPE_LOONGARCH['R_LARCH_64']: _RELOCATION_RECIPE_TYPE(bytesize=8, has_addend=True, calc_func=_reloc_calc_sym_plus_addend), ENUM_RELOC_TYPE_LOONGARCH['R_LARCH_ADD8']: _RELOCATION_RECIPE_TYPE(bytesize=1, has_addend=True, calc_func=_reloc_calc_sym_plus_value), ENUM_RELOC_TYPE_LOONGARCH['R_LARCH_SUB8']: _RELOCATION_RECIPE_TYPE(bytesize=1, has_addend=True, calc_func=_reloc_calc_value_minus_sym_addend), ENUM_RELOC_TYPE_LOONGARCH['R_LARCH_ADD16']: _RELOCATION_RECIPE_TYPE(bytesize=2, has_addend=True, calc_func=_reloc_calc_sym_plus_value), ENUM_RELOC_TYPE_LOONGARCH['R_LARCH_SUB16']: _RELOCATION_RECIPE_TYPE(bytesize=2, has_addend=True, calc_func=_reloc_calc_value_minus_sym_addend), ENUM_RELOC_TYPE_LOONGARCH['R_LARCH_ADD32']: _RELOCATION_RECIPE_TYPE(bytesize=4, has_addend=True, calc_func=_reloc_calc_sym_plus_value), ENUM_RELOC_TYPE_LOONGARCH['R_LARCH_SUB32']: _RELOCATION_RECIPE_TYPE(bytesize=4, has_addend=True, calc_func=_reloc_calc_value_minus_sym_addend), ENUM_RELOC_TYPE_LOONGARCH['R_LARCH_ADD64']: _RELOCATION_RECIPE_TYPE(bytesize=8, has_addend=True, calc_func=_reloc_calc_sym_plus_value), ENUM_RELOC_TYPE_LOONGARCH['R_LARCH_SUB64']: _RELOCATION_RECIPE_TYPE(bytesize=8, has_addend=True, calc_func=_reloc_calc_value_minus_sym_addend), ENUM_RELOC_TYPE_LOONGARCH['R_LARCH_32_PCREL']: _RELOCATION_RECIPE_TYPE(bytesize=4, has_addend=True, calc_func=_reloc_calc_sym_plus_addend_pcrel), ENUM_RELOC_TYPE_LOONGARCH['R_LARCH_64_PCREL']: _RELOCATION_RECIPE_TYPE(bytesize=8, has_addend=True, calc_func=_reloc_calc_sym_plus_addend_pcrel)}
    _RELOCATION_RECIPES_S390X = {ENUM_RELOC_TYPE_S390X['R_390_32']: _RELOCATION_RECIPE_TYPE(bytesize=4, has_addend=True, calc_func=_reloc_calc_sym_plus_addend), ENUM_RELOC_TYPE_S390X['R_390_PC32']: _RELOCATION_RECIPE_TYPE(bytesize=4, has_addend=True, calc_func=_reloc_calc_sym_plus_addend_pcrel), ENUM_RELOC_TYPE_S390X['R_390_64']: _RELOCATION_RECIPE_TYPE(bytesize=8, has_addend=True, calc_func=_reloc_calc_sym_plus_addend)}