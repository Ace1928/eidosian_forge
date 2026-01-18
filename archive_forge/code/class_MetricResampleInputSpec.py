import os
from ..base import TraitedSpec, File, traits, CommandLineInputSpec
from .base import WBCommand
from ... import logging
class MetricResampleInputSpec(CommandLineInputSpec):
    in_file = File(exists=True, mandatory=True, argstr='%s', position=0, desc='The metric file to resample')
    current_sphere = File(exists=True, mandatory=True, argstr='%s', position=1, desc='A sphere surface with the mesh that the metric is currently on')
    new_sphere = File(exists=True, mandatory=True, argstr='%s', position=2, desc='A sphere surface that is in register with <current-sphere> and has the desired output mesh')
    method = traits.Enum('ADAP_BARY_AREA', 'BARYCENTRIC', argstr='%s', mandatory=True, position=3, desc='The method name - ADAP_BARY_AREA method is recommended for ordinary metric data, because it should use all data while downsampling, unlike BARYCENTRIC. If ADAP_BARY_AREA is used, exactly one of area_surfs or area_metrics must be specified')
    out_file = File(name_source=['new_sphere'], name_template='%s.out', keep_extension=True, argstr='%s', position=4, desc='The output metric')
    area_surfs = traits.Bool(position=5, argstr='-area-surfs', xor=['area_metrics'], desc='Specify surfaces to do vertex area correction based on')
    area_metrics = traits.Bool(position=5, argstr='-area-metrics', xor=['area_surfs'], desc='Specify vertex area metrics to do area correction based on')
    current_area = File(exists=True, position=6, argstr='%s', desc='A relevant anatomical surface with <current-sphere> mesh OR a metric file with vertex areas for <current-sphere> mesh')
    new_area = File(exists=True, position=7, argstr='%s', desc='A relevant anatomical surface with <current-sphere> mesh OR a metric file with vertex areas for <current-sphere> mesh')
    roi_metric = File(exists=True, position=8, argstr='-current-roi %s', desc='Input roi on the current mesh used to exclude non-data vertices')
    valid_roi_out = traits.Bool(position=9, argstr='-valid-roi-out', desc='Output the ROI of vertices that got data from valid source vertices')
    largest = traits.Bool(position=10, argstr='-largest', desc='Use only the value of the vertex with the largest weight')