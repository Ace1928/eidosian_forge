from ..base import CommandLineInputSpec, CommandLine, TraitedSpec, File
class VtoMat(CommandLine):
    """
    Convert a nifti file into a vista file.

    Example
    -------
    >>> vimage = VtoMat()
    >>> vimage.inputs.in_file = 'image.v'
    >>> vimage.cmdline
    'vtomat -in image.v -out image.mat'
    >>> vimage.run()  # doctest: +SKIP

    """
    _cmd = 'vtomat'
    input_spec = VtoMatInputSpec
    output_spec = VtoMatOutputSpec