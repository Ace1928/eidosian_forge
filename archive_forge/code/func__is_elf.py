import os
import shutil
import subprocess
import sys
def _is_elf(filename):
    """Return True if the given file is an ELF file"""
    elf_header = b'\x7fELF'
    try:
        with open(filename, 'br') as thefile:
            return thefile.read(4) == elf_header
    except FileNotFoundError:
        return False