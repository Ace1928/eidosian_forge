import os
import glob
from ...utils.filemanip import split_filename
from ..base import (
class AnalyzeHeaderInputSpec(StdOutCommandLineInputSpec):
    in_file = File(exists=True, argstr='< %s', mandatory=True, position=1, desc='Tensor-fitted data filename')
    scheme_file = File(exists=True, argstr='%s', position=2, desc='Camino scheme file (b values / vectors, see camino.fsl2scheme)')
    readheader = File(exists=True, argstr='-readheader %s', position=3, desc='Reads header information from file and prints to stdout. If this option is not specified, then the program writes a header based on the other arguments.')
    printimagedims = File(exists=True, argstr='-printimagedims %s', position=3, desc='Prints image data and voxel dimensions as Camino arguments and exits.')
    printprogargs = File(exists=True, argstr='-printprogargs %s', position=3, desc='Prints data dimension (and type, if relevant) arguments for a specific Camino program, where prog is one of shredder, scanner2voxel, vcthreshselect, pdview, track.')
    printintelbyteorder = File(exists=True, argstr='-printintelbyteorder %s', position=3, desc='Prints 1 if the header is little-endian, 0 otherwise.')
    printbigendian = File(exists=True, argstr='-printbigendian %s', position=3, desc='Prints 1 if the header is big-endian, 0 otherwise.')
    initfromheader = File(exists=True, argstr='-initfromheader %s', position=3, desc='Reads header information from file and initializes a new header with the values read from the file. You may replace any combination of fields in the new header by specifying subsequent options.')
    data_dims = traits.List(traits.Int, desc='data dimensions in voxels', argstr='-datadims %s', minlen=3, maxlen=3, units='voxels')
    voxel_dims = traits.List(traits.Float, desc='voxel dimensions in mm', argstr='-voxeldims %s', minlen=3, maxlen=3, units='mm')
    centre = traits.List(traits.Int, argstr='-centre %s', minlen=3, maxlen=3, units='mm', desc='Voxel specifying origin of Talairach coordinate system for SPM, default [0 0 0].')
    picoseed = traits.List(traits.Int, argstr='-picoseed %s', minlen=3, maxlen=3, desc='Voxel specifying the seed (for PICo maps), default [0 0 0].', units='mm')
    nimages = traits.Int(argstr='-nimages %d', units='NA', desc='Number of images in the img file. Default 1.')
    datatype = traits.Enum('byte', 'char', '[u]short', '[u]int', 'float', 'complex', 'double', argstr='-datatype %s', desc='The char datatype is 8 bit (not the 16 bit char of Java), as specified by the Analyze 7.5 standard. The byte, ushort and uint types are not part of the Analyze specification but are supported by SPM.', mandatory=True)
    offset = traits.Int(argstr='-offset %d', units='NA', desc='According to the Analyze 7.5 standard, this is the byte offset in the .img file at which voxels start. This value can be negative to specify that the absolute value is applied for every image in the file.')
    greylevels = traits.List(traits.Int, argstr='-gl %s', minlen=2, maxlen=2, desc='Minimum and maximum greylevels. Stored as shorts in the header.', units='NA')
    scaleslope = traits.Float(argstr='-scaleslope %d', units='NA', desc='Intensities in the image are scaled by this factor by SPM and MRICro. Default is 1.0.')
    scaleinter = traits.Float(argstr='-scaleinter %d', units='NA', desc='Constant to add to the image intensities. Used by SPM and MRIcro.')
    description = traits.String(argstr='-description %s', desc='Short description - No spaces, max length 79 bytes. Will be null terminated automatically.')
    intelbyteorder = traits.Bool(argstr='-intelbyteorder', desc='Write header in intel byte order (little-endian).')
    networkbyteorder = traits.Bool(argstr='-networkbyteorder', desc='Write header in network byte order (big-endian). This is the default for new headers.')