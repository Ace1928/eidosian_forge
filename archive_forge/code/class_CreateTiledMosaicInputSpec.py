import os
from ..base import TraitedSpec, File, traits
from .base import ANTSCommand, ANTSCommandInputSpec
class CreateTiledMosaicInputSpec(ANTSCommandInputSpec):
    input_image = File(argstr='-i %s', exists=True, desc='Main input is a 3-D grayscale image.', mandatory=True)
    rgb_image = File(argstr='-r %s', exists=True, desc='An optional Rgb image can be added as an overlay.It must have the same imagegeometry as the input grayscale image.', mandatory=True)
    mask_image = File(argstr='-x %s', exists=True, desc='Specifies the ROI of the RGB voxels used.')
    alpha_value = traits.Float(argstr='-a %.2f', desc='If an Rgb image is provided, render the overlay using the specified alpha parameter.')
    output_image = traits.Str('output.png', argstr='-o %s', desc='The output consists of the tiled mosaic image.', usedefault=True)
    tile_geometry = traits.Str(argstr='-t %s', desc='The tile geometry specifies the number of rows and columnsin the output image. For example, if the user specifies "5x10", then 5 rows by 10 columns of slices are rendered. If R < 0 and C > 0 (or vice versa), the negative value is selectedbased on direction.')
    direction = traits.Int(argstr='-d %d', desc='Specifies the direction of the slices. If no direction is specified, the direction with the coarsest spacing is chosen.')
    pad_or_crop = traits.Str(argstr='-p %s', desc='argument passed to -p flag:[padVoxelWidth,<constantValue=0>][lowerPadding[0]xlowerPadding[1],upperPadding[0]xupperPadding[1],constantValue]The user can specify whether to pad or crop a specified voxel-width boundary of each individual slice. For this program, cropping is simply padding with negative voxel-widths.If one pads (+), the user can also specify a constant pad value (default = 0). If a mask is specified, the user can use the mask to define the region, by using the keyword "mask" plus an offset, e.g. "-p mask+3".')
    slices = traits.Str(argstr='-s %s', desc='Number of slices to increment Slice1xSlice2xSlice3[numberOfSlicesToIncrement,<minSlice=0>,<maxSlice=lastSlice>]')
    flip_slice = traits.Str(argstr='-f %s', desc='flipXxflipY')
    permute_axes = traits.Bool(argstr='-g', desc='doPermute')