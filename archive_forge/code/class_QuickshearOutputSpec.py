from .base import CommandLineInputSpec, CommandLine, traits, TraitedSpec, File
from ..external.due import BibTeX
class QuickshearOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='defaced output image')