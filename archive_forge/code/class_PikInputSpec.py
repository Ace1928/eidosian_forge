import glob
import os
import os.path
import re
import warnings
from ..base import (
from .base import aggregate_filename
class PikInputSpec(CommandLineInputSpec):
    input_file = File(desc='input file', exists=True, mandatory=True, argstr='%s', position=-2)
    _xor_image_type = ('jpg', 'png')
    jpg = traits.Bool(desc='Output a jpg file.', xor=_xor_image_type)
    png = traits.Bool(desc='Output a png file (default).', xor=_xor_image_type)
    output_file = File(desc='output file', argstr='%s', genfile=True, position=-1, name_source=['input_file'], hash_files=False, name_template='%s.png', keep_extension=False)
    clobber = traits.Bool(desc='Overwrite existing file.', argstr='-clobber', usedefault=True, default_value=True)
    scale = traits.Int(2, usedefault=True, desc='Scaling factor for resulting image. By default images areoutput at twice their original resolution.', argstr='--scale %s')
    width = traits.Int(desc='Autoscale the resulting image to have a fixed image width (in pixels).', argstr='--width %s')
    depth = traits.Enum(8, 16, desc='Bitdepth for resulting image 8 or 16 (MSB machines only!)', argstr='--depth %s')
    _xor_title = ('title_string', 'title_with_filename')
    title = traits.Either(traits.Bool(desc='Use input filename as title in resulting image.'), traits.Str(desc='Add a title to the resulting image.'), argstr='%s')
    title_size = traits.Int(desc='Font point size for the title.', argstr='--title_size %s', requires=['title'])
    annotated_bar = traits.Bool(desc='create an annotated bar to match the image (use height of the output image)', argstr='--anot_bar')
    minc_range = traits.Tuple(traits.Float, traits.Float, desc='Valid range of values for MINC file.', argstr='--range %s %s')
    _xor_image_range = ('image_range', 'auto_range')
    image_range = traits.Tuple(traits.Float, traits.Float, desc='Range of image values to use for pixel intensity.', argstr='--image_range %s %s', xor=_xor_image_range)
    auto_range = traits.Bool(desc='Automatically determine image range using a 5 and 95% PcT. (histogram)', argstr='--auto_range', xor=_xor_image_range)
    start = traits.Int(desc='Slice number to get. (note this is in voxel coordinates).', argstr='--slice %s')
    _xor_slice = ('slice_z', 'slice_y', 'slice_x')
    slice_z = traits.Bool(desc='Get an axial/transverse (z) slice.', argstr='-z', xor=_xor_slice)
    slice_y = traits.Bool(desc='Get a coronal (y) slice.', argstr='-y', xor=_xor_slice)
    slice_x = traits.Bool(desc='Get a sagittal (x) slice.', argstr='-x', xor=_xor_slice)
    triplanar = traits.Bool(desc='Create a triplanar view of the input file.', argstr='--triplanar')
    tile_size = traits.Int(desc='Pixel size for each image in a triplanar.', argstr='--tilesize %s')
    _xor_sagittal_offset = ('sagittal_offset', 'sagittal_offset_perc')
    sagittal_offset = traits.Int(desc='Offset the sagittal slice from the centre.', argstr='--sagittal_offset %s')
    sagittal_offset_perc = traits.Range(low=0, high=100, desc='Offset the sagittal slice by a percentage from the centre.', argstr='--sagittal_offset_perc %d')
    _xor_vertical_horizontal = ('vertical_triplanar_view', 'horizontal_triplanar_view')
    vertical_triplanar_view = traits.Bool(desc='Create a vertical triplanar view (Default).', argstr='--vertical', xor=_xor_vertical_horizontal)
    horizontal_triplanar_view = traits.Bool(desc='Create a horizontal triplanar view.', argstr='--horizontal', xor=_xor_vertical_horizontal)
    lookup = traits.Str(desc='Arguments to pass to minclookup', argstr='--lookup %s')