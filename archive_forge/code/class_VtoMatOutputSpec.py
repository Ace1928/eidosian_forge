from ..base import CommandLineInputSpec, CommandLine, TraitedSpec, File
class VtoMatOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='Output mat file')