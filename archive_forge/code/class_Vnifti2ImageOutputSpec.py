from ..base import CommandLineInputSpec, CommandLine, TraitedSpec, File
class Vnifti2ImageOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='Output vista file')