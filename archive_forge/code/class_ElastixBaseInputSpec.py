from ... import logging
from ..base import CommandLineInputSpec, Directory, traits
class ElastixBaseInputSpec(CommandLineInputSpec):
    output_path = Directory('./', exists=True, mandatory=True, usedefault=True, argstr='-out %s', desc='output directory')
    num_threads = traits.Int(1, usedefault=True, argstr='-threads %01d', nohash=True, desc='set the maximum number of threads of elastix')